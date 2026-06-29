"""In-memory poker orchestrator for the live demo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import eval7

from pokerbot_3000.domain.models import (
    ActionType,
    BoardRecognitionSnapshot,
    BoardRecognitionStatus,
    ClientConnectionState,
    ClientId,
    ClientStatus,
    ClientStatusUpdate,
    EventType,
    ExternalInputResult,
    GameEvent,
    HumanActionInput,
    HumanTableTalkInput,
    ObservationReceipt,
    OperatorControlResult,
    PendingInput,
    PendingInputType,
    PlayerState,
    PlayerStatus,
    PlayerType,
    PokerAction,
    PrivateAgentState,
    PrivateCardObservation,
    PublicGameState,
    PublicTableObservation,
    ShowdownSnapshot,
    ShowdownStatus,
    SidePotSnapshot,
    Street,
)
from pokerbot_3000.orchestrator.agents import AgentProfile

if TYPE_CHECKING:
    from pokerbot_3000.domain.cards import Card
    from pokerbot_3000.ports.llm import AgentDecision

BOARD_COMPLETE_CARD_COUNT = 5
HOLE_CARD_COUNT = 2
MIN_PLAYERS_FOR_HAND = 2
MIN_NON_ALL_IN_PLAYERS_FOR_BETTING = 2
ACTION_SPEECH_BREAK = '<break time="0.8s" />'


@dataclass(frozen=True, slots=True)
class DemoDefaults:
    """Configurable demo values for the initial state."""

    hand_id_prefix: str = "hand"
    starting_stack: int = 2_000
    small_blind: int = 10
    big_blind: int = 20
    dealer_seat: int = 1
    active_player_seat: int = 1
    board_confidence_threshold: float = 0.75
    required_stable_board_samples: int = 2


class InMemoryOrchestrator:
    """Python-owned game engine that runs until external input is required."""

    _turn_order = (1, 2, 3)

    def __init__(self, defaults: DemoDefaults | None = None) -> None:
        """Initialize the orchestrator with a three-player demo table."""
        self._defaults = defaults or DemoDefaults()
        self._state = self._build_initial_state(self._defaults)
        self._private_states = self._build_private_states()
        self._client_statuses = self._build_client_statuses()
        self._agent_profiles = self._build_agent_profiles()
        self._events: list[GameEvent] = []
        self._last_valid_board_candidate: tuple[tuple[str, str], ...] | None = None
        self._acted_this_street: set[int] = set()
        self._last_aggressor_by_street: dict[Street, int] = {}
        self._showdown_reveals: dict[int, list[Card]] = {}
        self._all_in_runout = False
        self._pending_presentation_event_id: str | None = None
        self._append_event(
            EventType.SYSTEM,
            source="orchestrator",
            summary="Python orchestrator engine initialized and stopped.",
            payload={
                "starting_stack": self._defaults.starting_stack,
                "small_blind": self._defaults.small_blind,
                "big_blind": self._defaults.big_blind,
            },
        )

    def public_state(self) -> PublicGameState:
        """Return a copy of the public state safe for callers to serialize."""
        return self._state.model_copy(deep=True)

    def private_states(self) -> dict[str, PrivateAgentState]:
        """Return copies of private states keyed by agent id."""
        return {agent_id: state.model_copy(deep=True) for agent_id, state in self._private_states.items()}

    def client_statuses(self) -> dict[ClientId, ClientStatus]:
        """Return copies of client statuses keyed by client id."""
        return {client_id: status.model_copy(deep=True) for client_id, status in self._client_statuses.items()}

    def update_client_status(self, client_id: ClientId, update: ClientStatusUpdate) -> ClientStatus:
        """Record the latest connection status for one external client."""
        status = ClientStatus(
            client_id=client_id,
            connection=update.connection,
            status=update.status,
            detail=update.detail,
        )
        self._client_statuses[client_id] = status
        return status.model_copy(deep=True)

    def events(self, limit: int = 50) -> list[GameEvent]:
        """Return the most recent event-log entries."""
        return [event.model_copy(deep=True) for event in self._events[-limit:]]

    def event_count(self) -> int:
        """Return the current in-memory event count."""
        return len(self._events)

    def event_by_id(self, event_id: str) -> GameEvent | None:
        """Return one event by id when it is still in memory."""
        for event in self._events:
            if event.event_id == event_id:
                return event.model_copy(deep=True)
        return None

    def events_since(self, index: int) -> list[GameEvent]:
        """Return events appended after a known index."""
        return [event.model_copy(deep=True) for event in self._events[index:]]

    def needs_public_board_observation(self) -> bool:
        """Return whether the engine is currently blocked on public board recognition."""
        return self._is_waiting_for(PendingInputType.PUBLIC_BOARD_CARDS)

    def pending_agent_action(self) -> str | None:
        """Return the agent id whose Gemma decision is needed, if any."""
        waiting_for = self._state.waiting_for
        if waiting_for is None or waiting_for.type != PendingInputType.AGENT_ACTION:
            return None
        return waiting_for.agent_id

    def private_state_for_agent(self, agent_id: str) -> PrivateAgentState:
        """Return one agent private state for LLM decision input."""
        self._ensure_known_agent(agent_id)
        return self._private_states[agent_id].model_copy(deep=True)

    def start_game(self) -> OperatorControlResult:
        """Start a fresh demo session and begin the first hand."""
        event_start = len(self._events)
        if self._state.automation_status != "stopped":
            return OperatorControlResult(
                accepted=False,
                reason="Game is already started.",
                events=[],
                state=self.public_state(),
            )

        self._state = self._build_initial_state(self._defaults)
        self._private_states = self._build_private_states()
        self._reset_transient_hand_state()
        self._append_event(
            EventType.GAME_STARTED,
            source="dashboard",
            summary="Operator started a fresh poker session.",
            payload={"hand_number": self._state.hand_number},
        )
        self._begin_hand(self._defaults.dealer_seat)
        return OperatorControlResult(
            accepted=True,
            reason="Game started; preflop action is open.",
            events=self.events_since(event_start),
            state=self.public_state(),
        )

    def stop_game(self) -> OperatorControlResult:
        """Stop orchestration and clear pending input."""
        event_start = len(self._events)
        if self._state.automation_status == "stopped":
            return OperatorControlResult(
                accepted=False,
                reason="Game is already stopped.",
                events=[],
                state=self.public_state(),
            )

        self._state.automation_status = "stopped"
        self._state.waiting_for = None
        self._state.legal_actions = []
        self._state.active_to_call = 0
        self._state.board_recognition = self._empty_board_recognition()
        self._state.showdown = ShowdownSnapshot()
        self._reset_transient_hand_state(clear_private=False)
        self._append_event(
            EventType.GAME_STOPPED,
            source="dashboard",
            summary="Operator stopped the game.",
            payload={"hand_id": self._state.hand_id},
        )
        return OperatorControlResult(
            accepted=True,
            reason="Game stopped by operator.",
            events=self.events_since(event_start),
            state=self.public_state(),
        )

    def submit_human_action(self, request: HumanActionInput) -> ExternalInputResult:
        """Consume human input and continue orchestration until blocked again."""
        event_start = len(self._events)
        if self._state.automation_status == "stopped":
            return self._external_rejection("Game is stopped. Start the game before submitting human input.")
        if not self._is_waiting_for(PendingInputType.HUMAN_ACTION, seat=request.seat):
            return self._external_rejection("The engine is not waiting for a human action from that seat.")

        if request.source == "voice":
            self._append_event(
                EventType.ACTION_PROPOSED,
                source=request.source,
                summary=f"S{request.seat} proposed {request.action.type} by voice.",
                payload={
                    "seat": request.seat,
                    "action": request.action.model_dump(mode="json"),
                    "raw_transcript": request.raw_transcript,
                    "confidence": request.confidence,
                },
            )
        try:
            self._commit_action(request.seat, request.action, source=request.source)
        except ValueError as exc:
            return ExternalInputResult(
                accepted=False,
                reason=str(exc),
                events=self.events_since(event_start),
                state=self.public_state(),
            )
        self._advance_after_action()
        return ExternalInputResult(
            accepted=True,
            reason="Human action consumed; engine advanced until the next external input.",
            events=self.events_since(event_start),
            state=self.public_state(),
        )

    def submit_human_table_talk(
        self,
        request: HumanTableTalkInput,
        *,
        speech: str | None,
        reaction: dict[str, object] | None = None,
        emotion: str = "calm",
    ) -> ExternalInputResult:
        """Record human table talk and queue a targeted agent response."""
        event_start = len(self._events)
        if self._state.automation_status == "stopped":
            return self._external_rejection("Game is stopped. Start the game before speaking to an agent.")
        if not self._is_waiting_for(PendingInputType.HUMAN_ACTION, seat=request.seat):
            return self._external_rejection("The engine is not waiting for human input from that seat.")
        self._ensure_known_agent(request.target_agent_id)

        self._append_event(
            EventType.HUMAN_TABLE_TALK,
            source=request.source,
            summary=f"S{request.seat} spoke to {self._agent_profiles[request.target_agent_id].display_name}.",
            payload={
                "seat": request.seat,
                "target_agent_id": request.target_agent_id,
                "message": request.message,
                "raw_transcript": request.raw_transcript,
                "confidence": request.confidence,
            },
        )
        if speech and speech.strip():
            self._queue_agent_presentation(
                request.target_agent_id,
                speech.strip(),
                reaction=reaction or {"intent": "table_talk_reply"},
                intent="table_talk_reply",
                presentation=(emotion, "nod"),
            )
        return ExternalInputResult(
            accepted=True,
            reason="Human table talk recorded; poker action is still pending.",
            events=self.events_since(event_start),
            state=self.public_state(),
        )

    def submit_agent_decision(self, decision: AgentDecision) -> ExternalInputResult:
        """Consume one Gemma agent decision and continue orchestration."""
        event_start = len(self._events)
        agent_id = self.pending_agent_action()
        if agent_id is None:
            return self._external_rejection("The engine is not waiting for an agent decision.")
        if decision.agent_id != agent_id:
            return self._external_rejection(f"The engine is waiting for {agent_id}, not {decision.agent_id}.")

        profile = self._agent_profiles[agent_id]
        try:
            self._commit_action(profile.seat, decision.action, source=f"agent:{agent_id}")
        except ValueError as exc:
            self.record_agent_decision_failed(agent_id, str(exc))
            return ExternalInputResult(
                accepted=False,
                reason=str(exc),
                events=self.events_since(event_start),
                state=self.public_state(),
            )

        self._append_event(
            EventType.AGENT_DECISION,
            source=f"agent:{agent_id}",
            summary=f"{profile.display_name} chose {decision.action.type}.",
            payload={
                "agent_id": agent_id,
                "action": decision.action.model_dump(mode="json"),
                "confidence": decision.confidence,
                "known_private_card_count": len(self._private_states[agent_id].hole_cards),
            },
        )
        presentation = self._queue_agent_presentation(
            agent_id,
            _agent_action_speech(profile.display_name, decision.action, decision.speech),
            reaction=decision.reaction,
            intent=_reaction_intent(decision.reaction),
            presentation=(_emotion_for_action(decision.action.type), _gesture_for_action(decision.action.type)),
        )
        self._pause_for_agent_presentation(agent_id, presentation.event_id)
        return ExternalInputResult(
            accepted=True,
            reason=f"{profile.display_name} agent decision consumed; waiting for presentation to finish.",
            events=self.events_since(event_start),
            state=self.public_state(),
        )

    def complete_presentation(self, event_id: str) -> ExternalInputResult:
        """Resume orchestration after an agent presentation has finished."""
        event_start = len(self._events)
        if self._pending_presentation_event_id != event_id:
            return self._external_rejection("The engine is not waiting for that presentation event.")
        self._pending_presentation_event_id = None
        self._advance_after_action()
        return ExternalInputResult(
            accepted=True,
            reason="Presentation complete; engine advanced until the next external input.",
            events=self.events_since(event_start),
            state=self.public_state(),
        )

    def record_agent_decision_failed(self, agent_id: str, message: str) -> GameEvent:
        """Record a failed Gemma decision while keeping the agent turn paused."""
        self._ensure_known_agent(agent_id)
        profile = self._agent_profiles[agent_id]
        self._state.waiting_for = PendingInput(
            type=PendingInputType.AGENT_ACTION,
            seat=profile.seat,
            agent_id=agent_id,
            client_id=profile.client_id,
            reason=f"Waiting for a valid Gemma decision from {profile.display_name}: {message}",
        )
        self._state.automation_status = "waiting_for_external_input"
        self._state.legal_actions = self._legal_actions()
        self._refresh_active_to_call()
        return self._append_event(
            EventType.AGENT_DECISION_FAILED,
            source=f"agent:{agent_id}",
            summary=f"Gemma decision failed for {profile.display_name}: {message}",
            payload={"agent_id": agent_id, "error": message},
        )

    def record_agent_banter_response(
        self,
        agent_id: str,
        speech: str,
        *,
        reaction: dict[str, object] | None = None,
        intent: str = "human_action_reaction",
        emotion: str = "calm",
    ) -> GameEvent:
        """Queue one targeted presentation response for an agent."""
        self._ensure_known_agent(agent_id)
        return self._queue_agent_presentation(
            agent_id,
            speech,
            reaction=reaction or {"intent": intent},
            intent=intent,
            presentation=(emotion, "nod"),
        )

    def record_public_observation(self, observation: PublicTableObservation) -> ObservationReceipt:
        """Record a public table observation from a camera bridge."""
        event_start = len(self._events)
        event = self._append_event(
            EventType.VISION_OBSERVATION,
            source=observation.source,
            summary=f"Detected {len(observation.board_cards)} public board card(s).",
            payload={
                **observation.model_dump(mode="json"),
                "expected_card_count": self._state.board_recognition.expected_card_count,
            },
        )
        if self.needs_public_board_observation():
            self._evaluate_public_board_observation(observation)
        return ObservationReceipt(
            accepted=True,
            reason=f"Recorded public observation; {len(self._events) - event_start} event(s) appended.",
            event=event,
        )

    def record_public_board_error(self, message: str, *, source: str = "public_board_loop") -> GameEvent:
        """Record a public board-recognition error without stopping the app."""
        if self.needs_public_board_observation():
            self._state.board_recognition = self._state.board_recognition.model_copy(
                update={"status": BoardRecognitionStatus.ERROR, "last_error": message, "stable_sample_count": 0},
            )
            self._last_valid_board_candidate = None
        return self._append_event(
            EventType.VISION_OBSERVATION,
            source=source,
            summary=f"Public board recognition error: {message}",
            payload={"error": message},
        )

    def record_client_private_cards(self, agent_id: str, observation: PrivateCardObservation) -> ExternalInputResult:
        """Consume thin-client private-card input and pause for that agent decision."""
        self._ensure_known_agent(agent_id)
        if self._state.automation_status == "stopped":
            return self._external_rejection("Game is stopped. Start the game before submitting private cards.")
        if observation.agent_id != agent_id:
            msg = f"Path agent_id {agent_id!r} does not match payload agent_id {observation.agent_id!r}."
            raise ValueError(msg)
        if not self._is_waiting_for(PendingInputType.PRIVATE_CARDS, agent_id=agent_id):
            return self._external_rejection(f"The engine is not waiting for private cards from {agent_id}.")

        event_start = len(self._events)
        validation_error = self._private_cards_validation_error(agent_id, observation)
        if validation_error is not None:
            self._append_event(
                EventType.PRIVATE_CARD_OBSERVATION,
                source=observation.source,
                summary=f"Rejected private card input from thin client {agent_id}: {validation_error}",
                payload={
                    "agent_id": agent_id,
                    "seat": observation.seat,
                    "hole_card_count": len(observation.hole_cards),
                    "source": observation.source,
                    "confidence": observation.confidence,
                    "error": validation_error,
                    "notes": observation.notes,
                },
            )
            return ExternalInputResult(
                accepted=False,
                reason=validation_error,
                events=self.events_since(event_start),
                state=self.public_state(),
            )
        self._store_private_cards(agent_id, observation)
        self._pause_for_agent_action(agent_id)
        return ExternalInputResult(
            accepted=True,
            reason=f"{agent_id} private cards consumed; waiting for Gemma decision.",
            events=self.events_since(event_start),
            state=self.public_state(),
        )

    def record_revealed_cards(self, seat: int, cards: list[Card], *, source: str) -> ExternalInputResult:
        """Consume one showdown reveal crop and continue runout or resolution."""
        if self._state.automation_status == "stopped":
            return self._external_rejection("Game is stopped. Start the game before submitting revealed cards.")
        if not self._is_waiting_for(PendingInputType.REVEALED_CARDS, seat=seat):
            return self._external_rejection(f"The engine is not waiting for revealed cards from seat {seat}.")

        event_start = len(self._events)
        error = self._revealed_cards_validation_error(seat, cards)
        if error is not None:
            self._state.showdown = self._state.showdown.model_copy(
                update={"status": ShowdownStatus.ERROR, "last_error": error},
            )
            self._append_event(
                EventType.REVEALED_CARD_OBSERVATION,
                source=source,
                summary=f"Rejected revealed cards for S{seat}: {error}",
                payload={"seat": seat, "error": error},
            )
            return ExternalInputResult(
                accepted=False,
                reason=error,
                events=self.events_since(event_start),
                state=self.public_state(),
            )

        self._showdown_reveals[seat] = cards
        self._state.showdown = self._state.showdown.model_copy(
            update={
                "status": ShowdownStatus.REVEALING,
                "revealed_cards_by_seat": self._showdown_reveals,
                "last_error": None,
            },
        )
        self._append_event(
            EventType.REVEALED_CARD_OBSERVATION,
            source=source,
            summary=f"Read revealed cards for S{seat}.",
            payload={"seat": seat, "hole_card_count": len(cards), "source": source},
        )
        self._continue_after_reveal()
        return ExternalInputResult(
            accepted=True,
            reason=f"Revealed cards for seat {seat} consumed.",
            events=self.events_since(event_start),
            state=self.public_state(),
        )

    def _begin_hand(self, dealer_seat: int) -> None:
        self._reset_transient_hand_state()
        active_seats = self._seats_with_chips()
        dealer = dealer_seat if dealer_seat in active_seats else active_seats[0]
        small_blind_seat, big_blind_seat = self._blind_seats_for(dealer, active_seats)

        self._state.hand_id = _hand_id(self._defaults.hand_id_prefix, self._state.hand_number)
        self._state.street = Street.PREFLOP
        self._state.dealer_seat = dealer
        self._state.small_blind_seat = small_blind_seat
        self._state.big_blind_seat = big_blind_seat
        self._state.small_blind = self._defaults.small_blind
        self._state.big_blind = self._defaults.big_blind
        self._state.board = []
        self._state.board_source = "manual"
        self._state.board_confidence = 0.0
        self._state.pot = 0
        self._state.current_bet_to_call = 0
        self._state.active_to_call = 0
        self._state.min_raise_to = self._defaults.big_blind
        self._state.side_pots = []
        self._state.waiting_for = None
        self._state.legal_actions = []
        self._state.board_recognition = self._empty_board_recognition()
        self._state.showdown = ShowdownSnapshot()
        self._private_states = self._build_private_states()
        for seat, player in self._state.players.items():
            player.committed_this_street = 0
            player.committed_this_hand = 0
            player.status = PlayerStatus.IN_HAND if seat in active_seats else PlayerStatus.OUT

        self._append_event(
            EventType.HAND_STARTED,
            source="orchestrator",
            summary=f"Hand {self._state.hand_number} started with S{dealer} as dealer.",
            payload={
                "hand_id": self._state.hand_id,
                "hand_number": self._state.hand_number,
                "dealer_seat": dealer,
                "small_blind_seat": small_blind_seat,
                "big_blind_seat": big_blind_seat,
            },
        )
        self._post_blind(small_blind_seat, self._defaults.small_blind, "small blind")
        self._post_blind(big_blind_seat, self._defaults.big_blind, "big blind")
        self._state.current_bet_to_call = max(
            self._state.players[small_blind_seat].committed_this_street,
            self._state.players[big_blind_seat].committed_this_street,
        )
        self._state.min_raise_to = self._state.current_bet_to_call + self._defaults.big_blind
        first_seat = self._next_action_seat_after(big_blind_seat)
        if first_seat is None:
            self._begin_showdown()
            return
        self._queue_orchestrator_speech(_hand_setup_speech(self._state, first_seat), intent="hand_setup")
        self._route_action_to(first_seat)

    def _reset_transient_hand_state(self, *, clear_private: bool = True) -> None:
        self._last_valid_board_candidate = None
        self._acted_this_street = set()
        self._last_aggressor_by_street = {}
        self._showdown_reveals = {}
        self._all_in_runout = False
        self._pending_presentation_event_id = None
        if clear_private:
            self._private_states = self._build_private_states()

    def _post_blind(self, seat: int, amount: int, label: str) -> None:
        player = self._state.players[seat]
        posted = min(player.stack, amount)
        player.stack -= posted
        player.committed_this_street += posted
        player.committed_this_hand += posted
        self._state.pot += posted
        if player.stack == 0:
            player.status = PlayerStatus.ALL_IN
        self._refresh_side_pots()
        self._append_event(
            EventType.BLIND_POSTED,
            source="orchestrator",
            summary=f"S{seat} posted {posted} chips for the {label}.",
            payload={"seat": seat, "amount": posted, "blind": label, "pot": self._state.pot},
        )

    def _advance_after_action(self) -> None:
        if self._award_if_uncontested():
            return
        if self._is_betting_round_complete():
            if self._should_enter_all_in_runout():
                self._begin_all_in_runout()
                return
            self._finish_betting_round()
            return
        self._continue_to_next_action()

    def _continue_to_next_action(self) -> None:
        next_seat = self._next_action_seat_after(self._state.active_player_seat)
        if next_seat is None:
            self._finish_betting_round()
            return
        self._route_action_to(next_seat)

    def _route_action_to(self, seat: int) -> None:
        self._set_active_player(seat)
        agent_id = self._agent_id_for_seat(seat)
        if agent_id is None:
            self._pause_for_human_action(seat)
            return
        if self._private_states[agent_id].hole_cards:
            self._pause_for_agent_action(agent_id)
            return
        self._pause_for_private_cards(agent_id)

    def _finish_betting_round(self) -> None:
        if len(self._state.board) < BOARD_COMPLETE_CARD_COUNT:
            expected_count = 3 if len(self._state.board) == 0 else len(self._state.board) + 1
            self._request_public_board_cards(expected_count)
            return
        self._begin_showdown()

    def _begin_all_in_runout(self) -> None:
        self._all_in_runout = True
        self._state.showdown = self._state.showdown.model_copy(update={"status": ShowdownStatus.REVEALING})
        self._begin_showdown_reveals()

    def _begin_showdown(self) -> None:
        if self._award_if_uncontested():
            return
        self._state.street = Street.SHOWDOWN
        self._begin_showdown_reveals()

    def _begin_showdown_reveals(self) -> None:
        reveal_order = self._showdown_reveal_order()
        self._state.showdown = ShowdownSnapshot(
            status=ShowdownStatus.REVEALING,
            reveal_order=reveal_order,
            revealed_cards_by_seat=self._showdown_reveals,
        )
        self._request_next_reveal()

    def _continue_after_reveal(self) -> None:
        if not self._all_required_reveals_read():
            self._request_next_reveal()
            return
        if len(self._state.board) < BOARD_COMPLETE_CARD_COUNT:
            self._state.showdown = self._state.showdown.model_copy(update={"status": ShowdownStatus.RUNNING_OUT})
            expected_count = 3 if len(self._state.board) == 0 else len(self._state.board) + 1
            self._request_public_board_cards(expected_count)
            return
        self._resolve_showdown()

    def _commit_action(self, seat: int, action: PokerAction, *, source: str) -> None:
        if seat != self._state.active_player_seat:
            msg = f"Seat {seat} is not the active player."
            raise ValueError(msg)
        self._validate_action(seat, action)

        player = self._state.players[seat]
        before_commitment = player.committed_this_street
        amount = self._chip_delta_for(seat, action)
        player.stack -= amount
        player.committed_this_street += amount
        player.committed_this_hand += amount
        self._state.pot += amount

        if action.type == ActionType.FOLD:
            player.status = PlayerStatus.FOLDED
        elif player.stack == 0:
            player.status = PlayerStatus.ALL_IN
        elif player.status == PlayerStatus.ACTIVE:
            player.status = PlayerStatus.IN_HAND

        aggressive = False
        if player.committed_this_street > self._state.current_bet_to_call:
            self._state.current_bet_to_call = player.committed_this_street
            self._state.min_raise_to = self._state.current_bet_to_call + self._defaults.big_blind
            aggressive = True
        if aggressive:
            self._last_aggressor_by_street[self._state.street] = seat
            self._acted_this_street = {seat}
        else:
            self._acted_this_street.add(seat)

        self._refresh_active_to_call()
        self._refresh_side_pots()
        amount_text = f" {amount}" if amount else ""
        self._append_event(
            EventType.ACTION_COMMITTED,
            source=source,
            summary=f"S{seat} committed {action.type}{amount_text}.",
            payload={
                "seat": seat,
                "action": action.model_dump(mode="json"),
                "delta": amount,
                "previous_commitment": before_commitment,
                "street_commitment": player.committed_this_street,
                "pot": self._state.pot,
                "stack": player.stack,
            },
        )

    def _validate_action(self, seat: int, action: PokerAction) -> None:  # noqa: C901
        player = self._state.players[seat]
        to_call = self._to_call_for(seat)
        if action.type not in self._legal_actions():
            msg = f"{action.type} is not legal for seat {seat}."
            raise ValueError(msg)
        if action.type == ActionType.CHECK and to_call:
            msg = f"Seat {seat} must call {to_call}, raise, or fold."
            raise ValueError(msg)
        if action.type == ActionType.CALL and to_call <= 0:
            msg = f"Seat {seat} has nothing to call."
            raise ValueError(msg)
        if action.type == ActionType.BET:
            target = action.amount or 0
            if self._state.current_bet_to_call:
                msg = "Cannot bet after a bet is already open."
                raise ValueError(msg)
            if target < self._defaults.big_blind and target < player.committed_this_street + player.stack:
                msg = f"Bet must be at least {self._defaults.big_blind} unless all-in."
                raise ValueError(msg)
            if target > player.committed_this_street + player.stack:
                msg = f"Seat {seat} does not have enough chips to bet {target}."
                raise ValueError(msg)
        if action.type == ActionType.RAISE_TO:
            target = action.amount or 0
            max_target = player.committed_this_street + player.stack
            if target <= self._state.current_bet_to_call:
                msg = f"Raise must exceed {self._state.current_bet_to_call}."
                raise ValueError(msg)
            if target < self._state.min_raise_to and target < max_target:
                msg = f"Raise must be to at least {self._state.min_raise_to} unless all-in."
                raise ValueError(msg)
            if target > max_target:
                msg = f"Seat {seat} does not have enough chips to raise to {target}."
                raise ValueError(msg)

    def _chip_delta_for(self, seat: int, action: PokerAction) -> int:
        player = self._state.players[seat]
        if action.type in {ActionType.BET, ActionType.RAISE_TO}:
            return max(0, (action.amount or 0) - player.committed_this_street)
        if action.type == ActionType.CALL:
            return min(self._to_call_for(seat), player.stack)
        if action.type == ActionType.ALL_IN:
            return player.stack
        return 0

    def _store_private_cards(self, agent_id: str, observation: PrivateCardObservation) -> None:
        self._private_states[agent_id] = PrivateAgentState(
            agent_id=agent_id,
            seat=observation.seat,
            hole_cards=observation.hole_cards,
            source=observation.source,
            confidence=observation.confidence,
            last_updated_at=datetime.now(UTC),
        )
        self._append_event(
            EventType.PRIVATE_CARD_OBSERVATION,
            source=observation.source,
            summary=f"Received private card input from thin client {agent_id}.",
            payload={
                "agent_id": agent_id,
                "seat": observation.seat,
                "hole_card_count": len(observation.hole_cards),
                "source": observation.source,
                "confidence": observation.confidence,
                "notes": observation.notes,
            },
        )

    def _evaluate_public_board_observation(self, observation: PublicTableObservation) -> None:
        expected_count = self._state.board_recognition.expected_card_count
        if expected_count is None:
            return

        validation_error = self._public_board_validation_error(observation, expected_count)
        if validation_error is not None:
            self._last_valid_board_candidate = None
            self._state.board_recognition = self._state.board_recognition.model_copy(
                update={
                    "status": BoardRecognitionStatus.DETECTING,
                    "latest_observation": observation,
                    "stable_sample_count": 0,
                    "last_error": validation_error,
                },
            )
            return

        candidate = _board_key(observation.board_cards)
        stable_count = 1
        if candidate == self._last_valid_board_candidate:
            stable_count = self._state.board_recognition.stable_sample_count + 1
        self._last_valid_board_candidate = candidate
        self._state.board_recognition = self._state.board_recognition.model_copy(
            update={
                "status": BoardRecognitionStatus.DETECTING,
                "latest_observation": observation,
                "stable_sample_count": stable_count,
                "last_error": None,
            },
        )

        if stable_count >= self._defaults.required_stable_board_samples:
            self._commit_public_board(observation)

    def _public_board_validation_error(self, observation: PublicTableObservation, expected_count: int) -> str | None:
        cards = observation.board_cards
        if len(cards) != expected_count:
            return f"Expected {expected_count} board cards, detected {len(cards)}."
        if observation.confidence < self._defaults.board_confidence_threshold:
            return f"Confidence {observation.confidence:.2f} is below {self._defaults.board_confidence_threshold:.2f}."
        if len(set(_board_key(cards))) != len(cards):
            return "Detected duplicate board cards."
        committed = _board_key(self._state.board)
        candidate_prefix = _board_key(cards[: len(self._state.board)])
        if committed != candidate_prefix:
            return "Previously committed board cards changed."
        return None

    def _private_cards_validation_error(self, agent_id: str, observation: PrivateCardObservation) -> str | None:
        profile = self._agent_profiles[agent_id]
        if observation.seat != profile.seat:
            return f"Expected private cards for S{profile.seat}, detected S{observation.seat}."
        if len(observation.hole_cards) != HOLE_CARD_COUNT:
            return f"Expected 2 private cards for {profile.display_name}, detected {len(observation.hole_cards)}."
        if len(set(_board_key(observation.hole_cards))) != len(observation.hole_cards):
            return "Detected duplicate private card."
        return None

    def _commit_public_board(self, observation: PublicTableObservation) -> None:
        cards = observation.board_cards
        self._state.board = cards
        self._state.board_source = observation.source
        self._state.board_confidence = observation.confidence
        self._state.street = _street_for_board_count(len(cards))
        self._last_valid_board_candidate = None
        self._append_event(
            EventType.BOARD_CARDS_COMMITTED,
            source="orchestrator",
            summary=f"Committed {self._state.street} board: {_format_board(cards)}.",
            payload={
                "board": [card.model_dump(mode="json") for card in cards],
                "street": self._state.street,
                "confidence": observation.confidence,
            },
        )

        if self._all_in_runout:
            if len(cards) < BOARD_COMPLETE_CARD_COUNT:
                self._request_public_board_cards(len(cards) + 1)
                return
            self._state.board_recognition = self._complete_board_recognition()
            self._resolve_showdown()
            return

        if len(cards) == BOARD_COMPLETE_CARD_COUNT:
            self._state.board_recognition = self._complete_board_recognition()
        self._begin_betting_round()

    def _begin_betting_round(self) -> None:
        for player in self._state.players.values():
            player.committed_this_street = 0
            if player.status == PlayerStatus.ACTIVE:
                player.status = PlayerStatus.IN_HAND
        self._state.current_bet_to_call = 0
        self._state.active_to_call = 0
        self._state.min_raise_to = self._defaults.big_blind
        self._acted_this_street = set()
        first_seat = self._first_live_after(self._state.dealer_seat)
        if first_seat is None:
            self._award_if_uncontested()
            return
        self._append_event(
            EventType.ENGINE_PAUSED,
            source="orchestrator",
            summary=f"Betting opened on the {self._state.street}.",
            payload={"street": self._state.street},
        )
        self._route_action_to(first_seat)

    def _request_public_board_cards(self, expected_count: int) -> None:
        stage_name = _stage_name_for_count(expected_count)
        self._state.waiting_for = PendingInput(
            type=PendingInputType.PUBLIC_BOARD_CARDS,
            reason=f"Waiting for a stable {stage_name} board observation.",
        )
        self._state.automation_status = "waiting_for_external_input"
        self._state.legal_actions = []
        self._state.active_to_call = 0
        self._state.board_recognition = BoardRecognitionSnapshot(
            status=BoardRecognitionStatus.WAITING,
            expected_card_count=expected_count,
            latest_observation=self._state.board_recognition.latest_observation,
            required_stable_samples=self._defaults.required_stable_board_samples,
            confidence_threshold=self._defaults.board_confidence_threshold,
            instruction=_instruction_for_count(expected_count),
        )
        self._append_event(
            EventType.ENGINE_PAUSED,
            source="orchestrator",
            summary=f"Engine paused for {stage_name} board recognition.",
            payload=self._state.waiting_for.model_dump(mode="json"),
        )
        self._queue_orchestrator_speech(
            _speech_for_count(expected_count, self._state.pot),
            intent=f"request_{stage_name}",
        )

    def _queue_orchestrator_speech(self, speech: str, *, intent: str) -> None:
        self._append_event(
            EventType.PRESENTATION_COMMAND,
            source="orchestrator",
            summary="Queued orchestrator speech.",
            payload={
                "target_client": ClientId.DASHBOARD,
                "voice": "orchestrator",
                "intent": intent,
                "speech": speech,
                "priority": "normal",
            },
        )

    def _queue_agent_presentation(
        self,
        agent_id: str,
        speech: str,
        *,
        reaction: dict[str, object],
        intent: str,
        presentation: tuple[str, str] = ("calm", "nod"),
    ) -> GameEvent:
        profile = self._agent_profiles[agent_id]
        emotion, gesture = presentation
        return self._append_event(
            EventType.PRESENTATION_COMMAND,
            source="orchestrator",
            summary=f"Queued {profile.display_name} presentation output.",
            payload={
                "target_client": profile.client_id,
                "intent": _reaction_intent(reaction) or intent,
                "speech": speech,
                "emotion": emotion,
                "gesture": gesture,
                "priority": "normal",
            },
        )

    def _queue_private_card_request(self, agent_id: str) -> GameEvent:
        profile = self._agent_profiles[agent_id]
        return self._append_event(
            EventType.PRESENTATION_COMMAND,
            source="orchestrator",
            summary=f"Queued {profile.display_name} private-card capture command.",
            payload={
                "target_client": profile.client_id,
                "intent": "request_private_cards",
                "speech": None,
                "emotion": "calm",
                "gesture": "look_down",
                "priority": "normal",
            },
        )

    def _pause_for_human_action(self, seat: int) -> None:
        self._state.waiting_for = PendingInput(
            type=PendingInputType.HUMAN_ACTION,
            seat=seat,
            reason="Waiting for the human player to declare an action.",
        )
        self._state.automation_status = "waiting_for_external_input"
        self._state.legal_actions = self._legal_actions()
        self._refresh_active_to_call()
        self._append_event(
            EventType.ENGINE_PAUSED,
            source="orchestrator",
            summary=f"Engine paused for human action at S{seat}.",
            payload=self._state.waiting_for.model_dump(mode="json"),
        )

    def _pause_for_private_cards(self, agent_id: str) -> None:
        profile = self._agent_profiles[agent_id]
        self._state.waiting_for = PendingInput(
            type=PendingInputType.PRIVATE_CARDS,
            seat=profile.seat,
            agent_id=agent_id,
            client_id=profile.client_id,
            reason=f"Waiting for {profile.display_name}'s thin client to capture private cards.",
        )
        self._state.automation_status = "waiting_for_external_input"
        self._state.legal_actions = []
        self._refresh_active_to_call()
        self._append_event(
            EventType.ENGINE_PAUSED,
            source="orchestrator",
            summary=f"Engine paused for {profile.display_name} private-card input.",
            payload=self._state.waiting_for.model_dump(mode="json"),
        )
        self._queue_orchestrator_speech(
            f"{profile.display_name}, check your cards.",
            intent="request_private_cards",
        )
        self._queue_private_card_request(agent_id)

    def _pause_for_agent_action(self, agent_id: str) -> None:
        profile = self._agent_profiles[agent_id]
        self._state.waiting_for = PendingInput(
            type=PendingInputType.AGENT_ACTION,
            seat=profile.seat,
            agent_id=agent_id,
            client_id=profile.client_id,
            reason=f"Waiting for Gemma to choose {profile.display_name}'s action.",
        )
        self._state.automation_status = "waiting_for_external_input"
        self._state.legal_actions = self._legal_actions()
        self._refresh_active_to_call()
        self._append_event(
            EventType.ENGINE_PAUSED,
            source="orchestrator",
            summary=f"Engine paused for {profile.display_name} Gemma decision.",
            payload=self._state.waiting_for.model_dump(mode="json"),
        )

    def _pause_for_agent_presentation(self, agent_id: str, event_id: str) -> None:
        profile = self._agent_profiles[agent_id]
        self._pending_presentation_event_id = event_id
        self._state.waiting_for = PendingInput(
            type=PendingInputType.PRESENTATION,
            seat=profile.seat,
            agent_id=agent_id,
            client_id=profile.client_id,
            reason=f"Waiting for {profile.display_name} to finish speaking.",
        )
        self._state.automation_status = "waiting_for_external_input"
        self._state.legal_actions = []
        self._state.active_to_call = 0
        self._append_event(
            EventType.ENGINE_PAUSED,
            source="orchestrator",
            summary=f"Engine paused for {profile.display_name} presentation.",
            payload={**self._state.waiting_for.model_dump(mode="json"), "presentation_event_id": event_id},
        )

    def _request_next_reveal(self) -> None:
        next_seat = next(
            (seat for seat in self._state.showdown.reveal_order if seat not in self._showdown_reveals),
            None,
        )
        if next_seat is None:
            self._continue_after_reveal()
            return
        self._state.waiting_for = PendingInput(
            type=PendingInputType.REVEALED_CARDS,
            seat=next_seat,
            reason=f"Waiting for S{next_seat} to reveal hole cards.",
        )
        self._state.automation_status = "waiting_for_external_input"
        self._state.legal_actions = []
        self._state.active_to_call = 0
        self._state.showdown = self._state.showdown.model_copy(
            update={
                "status": ShowdownStatus.REVEALING,
                "current_reveal_seat": next_seat,
                "revealed_cards_by_seat": self._showdown_reveals,
            },
        )
        self._append_event(
            EventType.ENGINE_PAUSED,
            source="orchestrator",
            summary=f"Engine paused for S{next_seat} revealed-card input.",
            payload=self._state.waiting_for.model_dump(mode="json"),
        )
        self._queue_orchestrator_speech(
            f"The pot is {self._state.pot}. {_player_name(self._state, next_seat)}, please reveal your cards.",
            intent="request_reveal",
        )

    def _append_event(
        self,
        event_type: EventType,
        *,
        source: str,
        summary: str,
        payload: dict[str, object],
    ) -> GameEvent:
        event = GameEvent(event_type=event_type, source=source, summary=summary, payload=payload)
        self._events.append(event)
        self._state.last_actions = [event.summary for event in self._events[-5:]]
        return event.model_copy(deep=True)

    def _external_rejection(self, reason: str) -> ExternalInputResult:
        return ExternalInputResult(accepted=False, reason=reason, events=[], state=self.public_state())

    def _is_waiting_for(
        self,
        pending_type: PendingInputType,
        *,
        seat: int | None = None,
        agent_id: str | None = None,
    ) -> bool:
        waiting_for = self._state.waiting_for
        if waiting_for is None or waiting_for.type != pending_type:
            return False
        if seat is not None and waiting_for.seat != seat:
            return False
        return not (agent_id is not None and waiting_for.agent_id != agent_id)

    def _set_active_player(self, seat: int) -> None:
        self._state.active_player_seat = seat
        for player_seat, player in self._state.players.items():
            if player.status == PlayerStatus.ACTIVE:
                player.status = PlayerStatus.IN_HAND
            if player_seat == seat and player.status == PlayerStatus.IN_HAND:
                player.status = PlayerStatus.ACTIVE
        self._state.legal_actions = self._legal_actions()
        self._refresh_active_to_call()

    def _legal_actions(self) -> list[ActionType]:
        player = self._state.players[self._state.active_player_seat]
        if player.status == PlayerStatus.ALL_IN or player.stack == 0:
            return []
        if self._to_call_for(self._state.active_player_seat):
            return [ActionType.FOLD, ActionType.CALL, ActionType.RAISE_TO, ActionType.ALL_IN]
        return [ActionType.CHECK, ActionType.BET, ActionType.ALL_IN]

    def _to_call_for(self, seat: int) -> int:
        return max(0, self._state.current_bet_to_call - self._state.players[seat].committed_this_street)

    def _refresh_active_to_call(self) -> None:
        self._state.active_to_call = self._to_call_for(self._state.active_player_seat)

    def _refresh_side_pots(self) -> None:
        contributions = {
            seat: player.committed_this_hand
            for seat, player in self._state.players.items()
            if player.committed_this_hand > 0
        }
        self._state.side_pots = _side_pots_for(contributions, self._eligible_showdown_seats())

    def _live_seats(self) -> list[int]:
        return [
            seat
            for seat, player in self._state.players.items()
            if player.status not in {PlayerStatus.FOLDED, PlayerStatus.OUT}
        ]

    def _seats_with_chips(self) -> list[int]:
        return [seat for seat in self._turn_order if self._state.players[seat].stack > 0]

    def _eligible_showdown_seats(self) -> set[int]:
        return {
            seat
            for seat, player in self._state.players.items()
            if player.status not in {PlayerStatus.FOLDED, PlayerStatus.OUT}
        }

    def _actionable_seats(self) -> list[int]:
        return [
            seat
            for seat in self._live_seats()
            if self._state.players[seat].status != PlayerStatus.ALL_IN and self._state.players[seat].stack > 0
        ]

    def _first_live_after(self, seat: int) -> int | None:
        return self._first_matching_after(seat, set(self._live_seats()))

    def _next_live_after(self, seat: int) -> int | None:
        return self._first_matching_after(seat, set(self._seats_with_chips()))

    def _next_action_seat_after(self, seat: int) -> int | None:
        return self._first_matching_after(seat, set(self._actionable_seats()))

    def _first_matching_after(self, seat: int, candidates: set[int]) -> int | None:
        if not candidates:
            return None
        current = seat
        for _ in self._turn_order:
            current = self._next_seat_after(current)
            if current in candidates:
                return current
        return None

    def _blind_seats_for(self, dealer: int, active_seats: list[int]) -> tuple[int, int]:
        if len(active_seats) == MIN_PLAYERS_FOR_HAND:
            small_blind = dealer
            big_blind = next(seat for seat in active_seats if seat != dealer)
            return small_blind, big_blind
        small_blind = self._first_matching_after(dealer, set(active_seats)) or dealer
        big_blind = self._first_matching_after(small_blind, set(active_seats)) or small_blind
        return small_blind, big_blind

    def _is_betting_round_complete(self) -> bool:
        actionable = set(self._actionable_seats())
        if len(self._live_seats()) <= 1:
            return True
        if not actionable:
            return True
        if not actionable.issubset(self._acted_this_street):
            return False
        return all(
            self._state.players[seat].committed_this_street == self._state.current_bet_to_call
            for seat in actionable
        )

    def _should_enter_all_in_runout(self) -> bool:
        live_count = len(self._live_seats())
        has_all_in_player = any(
            self._state.players[seat].status == PlayerStatus.ALL_IN for seat in self._live_seats()
        )
        return (
            live_count > 1
            and has_all_in_player
            and len(self._actionable_seats()) < MIN_NON_ALL_IN_PLAYERS_FOR_BETTING
            and self._is_betting_round_complete()
        )

    def _award_if_uncontested(self) -> bool:
        live_seats = self._live_seats()
        if len(live_seats) != 1 or self._state.pot == 0:
            return False
        winner = live_seats[0]
        awarded = self._state.pot
        self._state.players[winner].stack += awarded
        self._state.pot = 0
        self._state.side_pots = []
        self._state.waiting_for = None
        self._state.legal_actions = []
        self._state.active_to_call = 0
        self._state.automation_status = "complete"
        self._state.showdown = ShowdownSnapshot(
            status=ShowdownStatus.COMPLETE,
            winner_seats=[winner],
            winning_hand="Uncontested pot",
            pot_awarded=awarded,
        )
        self._append_event(
            EventType.SHOWDOWN_RESOLVED,
            source="orchestrator",
            summary=f"S{winner} won {awarded} chips uncontested.",
            payload={"winner_seats": [winner], "pot_awarded": awarded, "winning_hand": "Uncontested pot"},
        )
        self._start_next_hand_if_possible()
        return True

    def _showdown_reveal_order(self) -> list[int]:
        live_seats = self._live_seats()
        first = self._last_aggressor_by_street.get(Street.RIVER) or self._first_live_after(self._state.dealer_seat)
        if first is None or first not in live_seats:
            return live_seats
        ordered = [first]
        current = first
        while len(ordered) < len(live_seats):
            current = self._next_seat_after(current)
            if current in live_seats:
                ordered.append(current)
        return ordered

    def _all_required_reveals_read(self) -> bool:
        return all(seat in self._showdown_reveals for seat in self._state.showdown.reveal_order)

    def _revealed_cards_validation_error(self, seat: int, cards: list[Card]) -> str | None:
        if seat not in self._state.showdown.reveal_order:
            return f"Seat {seat} is not required to reveal."
        if len(cards) != HOLE_CARD_COUNT:
            return f"Expected 2 revealed cards from seat {seat}, detected {len(cards)}."
        known_cards = [*self._state.board]
        for revealed_seat, revealed_cards in self._showdown_reveals.items():
            if revealed_seat != seat:
                known_cards.extend(revealed_cards)
        candidate_cards = [*known_cards, *cards]
        if len(set(_board_key(candidate_cards))) != len(candidate_cards):
            return "Detected duplicate card in showdown reveal."
        return None

    def _resolve_showdown(self) -> None:
        if len(self._state.board) != BOARD_COMPLETE_CARD_COUNT or not self._all_required_reveals_read():
            return
        scores = {
            seat: eval7.evaluate([*_eval7_cards(self._state.board), *_eval7_cards(cards)])
            for seat, cards in self._showdown_reveals.items()
        }
        side_pots = self._state.side_pots or [SidePotSnapshot(amount=self._state.pot, eligible_seats=list(scores))]
        winner_seats: set[int] = set()
        pot_awarded = 0
        best_overall = max(scores.values())
        for pot in side_pots:
            eligible = [seat for seat in pot.eligible_seats if seat in scores]
            if not eligible or pot.amount == 0:
                continue
            best_score = max(scores[seat] for seat in eligible)
            winners = sorted(seat for seat in eligible if scores[seat] == best_score)
            winner_seats.update(winners)
            share, remainder = divmod(pot.amount, len(winners))
            for index, seat in enumerate(winners):
                self._state.players[seat].stack += share + (1 if index < remainder else 0)
            pot_awarded += pot.amount

        self._state.pot = 0
        self._state.side_pots = []
        self._state.street = Street.SHOWDOWN
        self._state.waiting_for = None
        self._state.legal_actions = []
        self._state.active_to_call = 0
        self._state.automation_status = "complete"
        winning_hand = eval7.handtype(best_overall)
        self._state.showdown = self._state.showdown.model_copy(
            update={
                "status": ShowdownStatus.COMPLETE,
                "current_reveal_seat": None,
                "winner_seats": sorted(winner_seats),
                "winning_hand": winning_hand,
                "pot_awarded": pot_awarded,
                "last_error": None,
                "revealed_cards_by_seat": self._showdown_reveals,
            },
        )
        winners_text = ", ".join(f"S{seat}" for seat in sorted(winner_seats))
        self._append_event(
            EventType.SHOWDOWN_RESOLVED,
            source="orchestrator",
            summary=f"{winners_text} won {pot_awarded} chips with {winning_hand}.",
            payload={
                "winner_seats": sorted(winner_seats),
                "winning_hand": winning_hand,
                "pot_awarded": pot_awarded,
            },
        )
        self._start_next_hand_if_possible()

    def _start_next_hand_if_possible(self) -> None:
        if len(self._seats_with_chips()) < MIN_PLAYERS_FOR_HAND:
            return
        self._state.hand_number += 1
        self._begin_hand(self._next_live_after(self._state.dealer_seat) or self._state.dealer_seat)

    def _next_seat_after(self, seat: int) -> int:
        index = self._turn_order.index(seat)
        return self._turn_order[(index + 1) % len(self._turn_order)]

    def _agent_id_for_seat(self, seat: int) -> str | None:
        for agent_id, profile in self._agent_profiles.items():
            if profile.seat == seat:
                return agent_id
        return None

    def _ensure_known_agent(self, agent_id: str) -> None:
        if agent_id not in self._private_states:
            msg = f"Unknown private agent {agent_id!r}."
            raise ValueError(msg)

    def _empty_board_recognition(self) -> BoardRecognitionSnapshot:
        return BoardRecognitionSnapshot(
            confidence_threshold=self._defaults.board_confidence_threshold,
            required_stable_samples=self._defaults.required_stable_board_samples,
        )

    def _complete_board_recognition(self) -> BoardRecognitionSnapshot:
        return self._state.board_recognition.model_copy(
            update={
                "status": BoardRecognitionStatus.COMPLETE,
                "expected_card_count": None,
                "stable_sample_count": 0,
                "last_error": None,
                "instruction": "Board recognition complete.",
            },
        )

    @staticmethod
    def _build_initial_state(defaults: DemoDefaults) -> PublicGameState:
        players = {
            1: PlayerState(
                name="Che",
                type=PlayerType.HUMAN,
                status=PlayerStatus.IN_HAND,
                stack=defaults.starting_stack,
            ),
            2: PlayerState(
                name="Reachy",
                type=PlayerType.ROBOT_AGENT,
                status=PlayerStatus.IN_HAND,
                stack=defaults.starting_stack,
            ),
            3: PlayerState(
                name="Eliza",
                type=PlayerType.WEB_AGENT,
                status=PlayerStatus.IN_HAND,
                stack=defaults.starting_stack,
            ),
        }
        return PublicGameState(
            hand_id=_hand_id(defaults.hand_id_prefix, 1),
            hand_number=1,
            street=Street.PREFLOP,
            dealer_seat=defaults.dealer_seat,
            small_blind=defaults.small_blind,
            big_blind=defaults.big_blind,
            active_player_seat=defaults.active_player_seat,
            min_raise_to=defaults.big_blind,
            players=players,
            legal_actions=[],
            automation_status="stopped",
            waiting_for=None,
            board_recognition=BoardRecognitionSnapshot(
                confidence_threshold=defaults.board_confidence_threshold,
                required_stable_samples=defaults.required_stable_board_samples,
            ),
            uncertainties=[],
        )

    @staticmethod
    def _build_private_states() -> dict[str, PrivateAgentState]:
        return {
            "reachy": PrivateAgentState(agent_id="reachy", seat=2),
            "eliza": PrivateAgentState(agent_id="eliza", seat=3),
        }

    @staticmethod
    def _build_agent_profiles() -> dict[str, AgentProfile]:
        return {
            "reachy": AgentProfile(
                agent_id="reachy",
                seat=2,
                client_id=ClientId.REACHY,
                display_name="Reachy",
                personality="playful physical robot poker player",
            ),
            "eliza": AgentProfile(
                agent_id="eliza",
                seat=3,
                client_id=ClientId.ELIZA,
                display_name="Eliza",
                personality="browser-based poker friend",
            ),
        }

    @staticmethod
    def _build_client_statuses() -> dict[ClientId, ClientStatus]:
        return {
            ClientId.DASHBOARD: ClientStatus(
                client_id=ClientId.DASHBOARD,
                connection=ClientConnectionState.READY,
                status="operator surface",
            ),
            ClientId.REACHY: ClientStatus(
                client_id=ClientId.REACHY,
                connection=ClientConnectionState.DISCONNECTED,
                status="thin capture/output client pending",
            ),
            ClientId.ELIZA: ClientStatus(
                client_id=ClientId.ELIZA,
                connection=ClientConnectionState.DISCONNECTED,
                status="thin browser capture/output client pending",
            ),
            ClientId.VOICE: ClientStatus(
                client_id=ClientId.VOICE,
                connection=ClientConnectionState.READY,
                status="server-side human voice worker",
            ),
            ClientId.PUBLIC_VISION: ClientStatus(
                client_id=ClientId.PUBLIC_VISION,
                connection=ClientConnectionState.DISCONNECTED,
                status="browser table-camera bridge pending",
            ),
        }


def _hand_id(prefix: str, hand_number: int) -> str:
    return f"{prefix}_{hand_number:03d}"


def _board_key(cards: list[Card]) -> tuple[tuple[str, str], ...]:
    return tuple((str(card.rank), str(card.suit)) for card in cards)


def _eval7_cards(cards: list[Card]) -> list[eval7.Card]:
    return [eval7.Card(f"{_rank_code(str(card.rank))}{_suit_code(str(card.suit))}") for card in cards]


def _rank_code(rank: str) -> str:
    return {"ace": "A", "king": "K", "queen": "Q", "jack": "J", "10": "T"}.get(rank, rank)


def _suit_code(suit: str) -> str:
    return {"spades": "s", "hearts": "h", "diamonds": "d", "clubs": "c"}[suit]


def _format_board(cards: list[Card]) -> str:
    return ", ".join(card.label for card in cards)


def _street_for_board_count(card_count: int) -> Street:
    return {0: Street.PREFLOP, 3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}[card_count]


def _stage_name_for_count(card_count: int) -> str:
    return {3: "flop", 4: "turn", 5: "river"}[card_count]


def _instruction_for_count(card_count: int) -> str:
    return {3: "Lay out the flop.", 4: "Reveal the turn.", 5: "Reveal the river."}[card_count]


def _speech_for_count(card_count: int, pot: int) -> str:
    lines = {
        3: f"The pot is {pot}. Please lay out the flop.",
        4: f"The pot is {pot}. Great, reveal the turn.",
        5: f"The pot is {pot}. Great, reveal the river.",
    }
    return lines[card_count]


def _hand_setup_speech(state: PublicGameState, first_seat: int) -> str:
    dealer = _player_name(state, state.dealer_seat)
    small_blind = _player_name(state, state.small_blind_seat or state.dealer_seat)
    big_blind = _player_name(state, state.big_blind_seat or state.dealer_seat)
    first = _player_name(state, first_seat)
    return (
        f"Move the dealer button to {dealer}. {small_blind} posts small blind {state.small_blind}. "
        f"{big_blind} posts big blind {state.big_blind}. Deal two cards. Action is on {first}."
    )


def _player_name(state: PublicGameState, seat: int) -> str:
    return state.players[seat].name


def _side_pots_for(contributions: dict[int, int], eligible_seats: set[int]) -> list[SidePotSnapshot]:
    pots: list[SidePotSnapshot] = []
    previous = 0
    for level in sorted(set(contributions.values())):
        contributors = [seat for seat, amount in contributions.items() if amount >= level]
        pot_amount = (level - previous) * len(contributors)
        eligible = sorted(seat for seat in contributors if seat in eligible_seats)
        if pot_amount > 0 and eligible:
            pots.append(SidePotSnapshot(amount=pot_amount, eligible_seats=eligible))
        previous = level
    return pots


def _reaction_intent(reaction: dict[str, object]) -> str:
    intent = reaction.get("intent")
    return intent if isinstance(intent, str) and intent else "announce_action"


def _agent_action_speech(display_name: str, action: PokerAction, speech: str | None) -> str:
    if speech and speech.strip():
        return speech.strip()
    return f"{_default_agent_remark(action)} {ACTION_SPEECH_BREAK} {_agent_action_declaration(display_name, action)}"


def _default_agent_remark(action: PokerAction) -> str:
    return {
        ActionType.CHECK: "Let me keep this steady.",
        ActionType.CALL: "That price is workable.",
        ActionType.BET: "Time to test the table.",
        ActionType.RAISE_TO: "I want a little more pressure here.",
        ActionType.ALL_IN: "No more half measures.",
        ActionType.FOLD: "This one can go.",
    }.get(action.type, "I have a move.")


def _agent_action_declaration(display_name: str, action: PokerAction) -> str:
    if action.amount is None:
        return f"{display_name} {action.type.replace('_', ' ')}."
    return f"{display_name} {action.type.replace('_', ' ')} {action.amount}."


def _emotion_for_action(action_type: ActionType) -> str:
    return {
        ActionType.CHECK: "calm",
        ActionType.CALL: "calm",
        ActionType.BET: "confident",
        ActionType.RAISE_TO: "confident",
        ActionType.ALL_IN: "celebrate",
        ActionType.FOLD: "sad",
    }.get(action_type, "confused")


def _gesture_for_action(action_type: ActionType) -> str:
    return {
        ActionType.CHECK: "nod",
        ActionType.CALL: "nod",
        ActionType.BET: "lean_in",
        ActionType.RAISE_TO: "lean_in",
        ActionType.ALL_IN: "big_nod",
        ActionType.FOLD: "look_down",
    }.get(action_type, "tilt")
