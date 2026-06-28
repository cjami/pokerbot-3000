"""In-memory orchestrator skeleton for the hackathon demo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pokerbot_3000.domain.models import (
    ActionType,
    BoardRecognitionSnapshot,
    BoardRecognitionStatus,
    ClientConnectionState,
    ClientId,
    ClientStatus,
    EventType,
    ExternalInputResult,
    GameEvent,
    HumanActionInput,
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
    Street,
)
from pokerbot_3000.orchestrator.agents import AgentProfile, StubPokerAgent

if TYPE_CHECKING:
    from pokerbot_3000.domain.cards import Card


@dataclass(frozen=True, slots=True)
class DemoDefaults:
    """Configurable demo values for the initial skeleton state."""

    hand_id: str = "hand_001"
    starting_stack: int = 2_000
    small_blind: int = 10
    big_blind: int = 20
    dealer_seat: int = 1
    active_player_seat: int = 1
    board_confidence_threshold: float = 0.75
    required_stable_board_samples: int = 2


class InMemoryOrchestrator:
    """Python-owned game engine that runs until external input is required."""

    _turn_order = (1, 3, 2)

    def __init__(self, defaults: DemoDefaults | None = None) -> None:
        """Initialize the orchestrator with a three-player demo hand."""
        self._defaults = defaults or DemoDefaults()
        self._state = self._build_initial_state(self._defaults)
        self._private_states = self._build_private_states()
        self._client_statuses = self._build_client_statuses()
        self._agent = StubPokerAgent()
        self._agent_profiles = self._build_agent_profiles()
        self._events: list[GameEvent] = []
        self._last_valid_board_candidate: tuple[tuple[str, str], ...] | None = None
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

    def needs_public_board_observation(self) -> bool:
        """Return whether the engine is currently blocked on public board recognition."""
        return self._is_waiting_for(PendingInputType.PUBLIC_BOARD_CARDS)

    def start_game(self) -> OperatorControlResult:
        """Start a fresh demo hand and run until the first external input."""
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
        self._last_valid_board_candidate = None
        self._append_event(
            EventType.GAME_STARTED,
            source="dashboard",
            summary="Operator started a fresh demo hand.",
            payload={"hand_id": self._state.hand_id},
        )
        self._set_active_player(self._defaults.active_player_seat)
        self._request_public_board_cards(3)
        return OperatorControlResult(
            accepted=True,
            reason="Game started; waiting for the flop.",
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
        self._state.board_recognition = BoardRecognitionSnapshot(
            confidence_threshold=self._defaults.board_confidence_threshold,
            required_stable_samples=self._defaults.required_stable_board_samples,
        )
        self._last_valid_board_candidate = None
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
        """Consume human input and continue automated orchestration until blocked again."""
        event_start = len(self._events)
        if self._state.automation_status == "stopped":
            return ExternalInputResult(
                accepted=False,
                reason="Game is stopped. Start the game before submitting human input.",
                events=[],
                state=self.public_state(),
            )
        if not self._is_waiting_for(PendingInputType.HUMAN_ACTION, seat=request.seat):
            return ExternalInputResult(
                accepted=False,
                reason="The engine is not waiting for a human action from that seat.",
                events=[],
                state=self.public_state(),
            )

        self._commit_action(request.seat, request.action, source=request.source)
        self._advance_until_blocked()
        return ExternalInputResult(
            accepted=True,
            reason="Human action consumed; engine advanced until the next external input.",
            events=self.events_since(event_start),
            state=self.public_state(),
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
                update={
                    "status": BoardRecognitionStatus.ERROR,
                    "last_error": message,
                    "stable_sample_count": 0,
                },
            )
            self._last_valid_board_candidate = None
        return self._append_event(
            EventType.VISION_OBSERVATION,
            source=source,
            summary=f"Public board recognition error: {message}",
            payload={"error": message},
        )

    def record_client_private_cards(
        self,
        agent_id: str,
        observation: PrivateCardObservation,
    ) -> ExternalInputResult:
        """Consume thin-client private-card input and run the agent turn internally."""
        self._ensure_known_agent(agent_id)
        if self._state.automation_status == "stopped":
            return ExternalInputResult(
                accepted=False,
                reason="Game is stopped. Start the game before submitting private cards.",
                events=[],
                state=self.public_state(),
            )
        if observation.agent_id != agent_id:
            msg = f"Path agent_id {agent_id!r} does not match payload agent_id {observation.agent_id!r}."
            raise ValueError(msg)
        if not self._is_waiting_for(PendingInputType.PRIVATE_CARDS, agent_id=agent_id):
            return ExternalInputResult(
                accepted=False,
                reason=f"The engine is not waiting for private cards from {agent_id}.",
                events=[],
                state=self.public_state(),
            )

        event_start = len(self._events)
        self._store_private_cards(agent_id, observation)
        self._run_agent_turn(agent_id)
        self._advance_until_blocked()
        return ExternalInputResult(
            accepted=True,
            reason=f"{agent_id} private cards consumed; internal agent acted.",
            events=self.events_since(event_start),
            state=self.public_state(),
        )

    def events_since(self, index: int) -> list[GameEvent]:
        """Return events appended after a known index."""
        return [event.model_copy(deep=True) for event in self._events[index:]]

    def _advance_until_blocked(self) -> None:
        next_seat = self._next_seat_after(self._state.active_player_seat)
        self._set_active_player(next_seat)
        agent_id = self._agent_id_for_seat(next_seat)
        if agent_id is None:
            self._pause_for_human_action(next_seat)
            return

        self._pause_for_private_cards(agent_id)

    def _run_agent_turn(self, agent_id: str) -> None:
        profile = self._agent_profiles[agent_id]
        private_state = self._private_states[agent_id]
        turn = self._agent.decide(profile, self.public_state(), private_state)
        self._append_event(
            EventType.AGENT_DECISION,
            source=f"agent:{agent_id}",
            summary=f"{profile.display_name} chose {turn.action.type}.",
            payload={
                "agent_id": agent_id,
                "action": turn.action.model_dump(mode="json"),
                "known_private_card_count": len(private_state.hole_cards),
            },
        )
        self._commit_action(profile.seat, turn.action, source=f"agent:{agent_id}")
        self._append_event(
            EventType.PRESENTATION_COMMAND,
            source="orchestrator",
            summary=f"Queued {profile.display_name} presentation output.",
            payload={
                "target_client": profile.client_id,
                "intent": turn.reaction,
                "speech": turn.speech,
                "priority": "normal",
            },
        )

    def _commit_action(self, seat: int, action: PokerAction, *, source: str) -> None:
        if seat != self._state.active_player_seat:
            msg = f"Seat {seat} is not the active player."
            raise ValueError(msg)

        player = self._state.players[seat]
        amount = self._chip_delta_for(action, player)
        if amount > player.stack:
            msg = f"Seat {seat} does not have enough chips for {amount}."
            raise ValueError(msg)

        player.stack -= amount
        player.committed_this_street += amount
        self._state.pot += amount
        if action.type in {ActionType.BET, ActionType.RAISE_TO}:
            self._state.current_bet_to_call = action.amount or 0
            self._state.min_raise_to = max(
                self._state.current_bet_to_call + self._defaults.big_blind,
                self._defaults.big_blind,
            )
        if action.type == ActionType.FOLD:
            player.status = PlayerStatus.FOLDED
        elif action.type == ActionType.ALL_IN:
            player.status = PlayerStatus.ALL_IN

        amount_text = f" {amount}" if amount else ""
        self._append_event(
            EventType.ACTION_COMMITTED,
            source=source,
            summary=f"S{seat} committed {action.type}{amount_text}.",
            payload={
                "seat": seat,
                "action": action.model_dump(mode="json"),
                "pot": self._state.pot,
                "stack": player.stack,
            },
        )

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

        cards = observation.board_cards
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

        candidate = _board_key(cards)
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

        next_expected = {3: 4, 4: 5}.get(len(cards))
        if next_expected is None:
            self._state.board_recognition = self._state.board_recognition.model_copy(
                update={
                    "status": BoardRecognitionStatus.COMPLETE,
                    "expected_card_count": None,
                    "stable_sample_count": 0,
                    "last_error": None,
                    "instruction": "Board recognition complete.",
                },
            )
            self._queue_orchestrator_speech("The board is complete. Che, action is on you.", intent="board_complete")
            self._pause_for_human_action(self._defaults.active_player_seat)
            return

        self._request_public_board_cards(next_expected)

    def _request_public_board_cards(self, expected_count: int) -> None:
        stage_name = _stage_name_for_count(expected_count)
        self._state.waiting_for = PendingInput(
            type=PendingInputType.PUBLIC_BOARD_CARDS,
            reason=f"Waiting for a stable {stage_name} board observation.",
        )
        self._state.automation_status = "waiting_for_external_input"
        self._state.legal_actions = []
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
        self._queue_orchestrator_speech(_speech_for_count(expected_count), intent=f"request_{stage_name}")

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

    def _pause_for_human_action(self, seat: int) -> None:
        self._state.waiting_for = PendingInput(
            type=PendingInputType.HUMAN_ACTION,
            seat=seat,
            reason="Waiting for the human player to declare an action.",
        )
        self._state.automation_status = "waiting_for_external_input"
        self._state.legal_actions = self._legal_actions()
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
        self._state.legal_actions = self._legal_actions()
        self._append_event(
            EventType.ENGINE_PAUSED,
            source="orchestrator",
            summary=f"Engine paused for {profile.display_name} private-card input.",
            payload=self._state.waiting_for.model_dump(mode="json"),
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

    def _chip_delta_for(self, action: PokerAction, player: PlayerState) -> int:
        if action.type in {ActionType.BET, ActionType.RAISE_TO}:
            return action.amount or 0
        if action.type == ActionType.CALL:
            return action.amount if action.amount is not None else self._state.current_bet_to_call
        if action.type == ActionType.ALL_IN:
            return player.stack
        return 0

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

    def _legal_actions(self) -> list[ActionType]:
        if self._state.current_bet_to_call:
            return [ActionType.FOLD, ActionType.CALL, ActionType.RAISE_TO]
        return [ActionType.CHECK, ActionType.BET]

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
            hand_id=defaults.hand_id,
            street=Street.PREFLOP,
            dealer_seat=defaults.dealer_seat,
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
            uncertainties=[
                "Rules engine is still a deterministic skeleton.",
                "Gemma-backed agent decisions are not wired yet.",
            ],
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
                connection=ClientConnectionState.DISCONNECTED,
                status="human voice input adapter pending",
            ),
            ClientId.PUBLIC_VISION: ClientStatus(
                client_id=ClientId.PUBLIC_VISION,
                connection=ClientConnectionState.DISCONNECTED,
                status="browser table-camera bridge pending",
            ),
        }


def _board_key(cards: list[Card]) -> tuple[tuple[str, str], ...]:
    return tuple((str(card.rank), str(card.suit)) for card in cards)


def _format_board(cards: list[Card]) -> str:
    return ", ".join(card.label for card in cards)


def _street_for_board_count(card_count: int) -> Street:
    streets = {
        0: Street.PREFLOP,
        3: Street.FLOP,
        4: Street.TURN,
        5: Street.RIVER,
    }
    return streets[card_count]


def _stage_name_for_count(card_count: int) -> str:
    stages = {
        3: "flop",
        4: "turn",
        5: "river",
    }
    return stages[card_count]


def _instruction_for_count(card_count: int) -> str:
    instructions = {
        3: "Lay out the flop.",
        4: "Reveal the turn.",
        5: "Reveal the river.",
    }
    return instructions[card_count]


def _speech_for_count(card_count: int) -> str:
    lines = {
        3: "Please lay out the flop.",
        4: "Great, reveal the turn.",
        5: "Great, reveal the river.",
    }
    return lines[card_count]
