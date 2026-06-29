import pytest

from pokerbot_3000.domain.cards import Card
from pokerbot_3000.domain.models import (
    EventType,
    ExternalInputResult,
    HumanActionInput,
    HumanTableTalkInput,
    ObservationReceipt,
    PendingInputType,
    PokerAction,
    PrivateCardObservation,
    PublicTableObservation,
    Street,
)
from pokerbot_3000.orchestrator import InMemoryOrchestrator
from pokerbot_3000.ports.llm import AgentDecision


def test_orchestrator_initializes_three_player_demo_state():
    orchestrator = InMemoryOrchestrator()

    state = orchestrator.public_state()

    assert state.hand_id == "hand_001"
    assert state.hand_number == 1
    assert len(state.players) == 3
    assert state.players[1].name == "Che"
    assert state.pot == 0
    assert state.automation_status == "stopped"
    assert state.waiting_for is None
    assert orchestrator.events()[0].event_type == "system"


def test_orchestrator_start_game_posts_blinds_and_pauses_for_preflop_human_action():
    orchestrator = InMemoryOrchestrator()

    result = orchestrator.start_game()

    assert result.accepted is True
    assert result.state.dealer_seat == 1
    assert result.state.small_blind_seat == 2
    assert result.state.big_blind_seat == 3
    assert result.state.pot == 30
    assert result.state.players[2].stack == 1990
    assert result.state.players[3].stack == 1980
    assert result.state.active_player_seat == 1
    assert result.state.active_to_call == 20
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.type == PendingInputType.HUMAN_ACTION
    assert "blind_posted" in {event.event_type for event in result.events}
    speech = [event.payload.get("speech") for event in result.events if event.event_type == "presentation_command"]
    assert speech == [
        "Move the dealer button to Che. Reachy posts small blind 10. Eliza posts big blind 20. "
        "Deal two cards. Action is on Che."
    ]


def test_orchestrator_stop_game_clears_pending_input():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()

    result = orchestrator.stop_game()

    assert result.accepted is True
    assert result.state.automation_status == "stopped"
    assert result.state.waiting_for is None
    assert result.state.legal_actions == []


def test_orchestrator_preflop_call_round_requests_flop_with_pot_aware_speech():
    orchestrator = InMemoryOrchestrator()

    result = _complete_preflop(orchestrator)

    assert result.state.pot == 60
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.type == PendingInputType.PUBLIC_BOARD_CARDS
    assert result.state.board_recognition.expected_card_count == 3
    speech = [event.payload.get("speech") for event in result.events if event.event_type == "presentation_command"]
    assert "The pot is 60. Please lay out the flop." in speech


def test_orchestrator_prompts_agents_to_check_private_cards():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()

    reachy_result = orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "call"}}))
    _record_private(orchestrator, "reachy")
    eliza_result = orchestrator.submit_agent_decision(_decision("reachy", "call"))

    reachy_speech = [
        event.payload.get("speech")
        for event in reachy_result.events
        if event.event_type == EventType.PRESENTATION_COMMAND and event.payload.get("voice") == "orchestrator"
    ]
    reachy_commands = [
        event.payload
        for event in reachy_result.events
        if event.event_type == EventType.PRESENTATION_COMMAND
        and event.payload.get("target_client") == "reachy"
        and event.payload.get("intent") == "request_private_cards"
    ]
    eliza_speech = [
        event.payload.get("speech")
        for event in eliza_result.events
        if event.event_type == EventType.PRESENTATION_COMMAND and event.payload.get("voice") == "orchestrator"
    ]
    eliza_commands = [
        event.payload
        for event in eliza_result.events
        if event.event_type == EventType.PRESENTATION_COMMAND
        and event.payload.get("target_client") == "eliza"
        and event.payload.get("intent") == "request_private_cards"
    ]
    assert "Reachy, check your cards." in reachy_speech
    assert len(reachy_commands) == 1
    assert "Eliza, check your cards." in eliza_speech
    assert len(eliza_commands) == 1


def test_orchestrator_commits_flop_then_pauses_for_first_postflop_agent_action():
    orchestrator = InMemoryOrchestrator()
    _complete_preflop(orchestrator)

    _record_public_observation(orchestrator, _flop())
    state = orchestrator.public_state()
    assert state.board == []
    assert state.board_recognition.stable_sample_count == 1

    _record_public_observation(orchestrator, _flop())
    state = orchestrator.public_state()

    assert state.street == Street.FLOP
    assert state.waiting_for is not None
    assert state.waiting_for.type == PendingInputType.AGENT_ACTION
    assert state.waiting_for.agent_id == "reachy"
    assert state.active_to_call == 0


def test_orchestrator_rejects_unstable_or_invalid_board_observations_when_waiting_for_board():
    orchestrator = InMemoryOrchestrator()
    _complete_preflop(orchestrator)

    _record_public_observation(orchestrator, _flop(), confidence=0.3)
    last_error = orchestrator.public_state().board_recognition.last_error
    assert last_error is not None
    assert last_error.startswith("Confidence")

    _record_public_observation(orchestrator, [_card("ace", "hearts"), _card("ace", "hearts"), _card("2", "clubs")])
    assert orchestrator.public_state().board_recognition.last_error == "Detected duplicate board cards."

    _record_public_observation(orchestrator, _flop())
    _record_public_observation(orchestrator, _flop())
    assert orchestrator.public_state().board_recognition.last_error is None


def test_orchestrator_records_public_observation_as_event_only_when_not_waiting_for_board():
    orchestrator = InMemoryOrchestrator()
    observation = PublicTableObservation(dealer_seat=1, board_cards=_flop(), street_hint=Street.FLOP, confidence=0.84)

    receipt = orchestrator.record_public_observation(observation)

    assert receipt.accepted is True
    assert receipt.event.event_type == "vision_observation"
    assert orchestrator.public_state().board == []


def test_orchestrator_private_cards_pause_for_gemma_agent_action_without_fallback():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()
    orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "call"}}))

    result = _record_private(orchestrator, "reachy")

    private_states = orchestrator.private_states()
    public_payload = orchestrator.public_state().model_dump()
    assert result.accepted is True
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.type == PendingInputType.AGENT_ACTION
    assert result.state.waiting_for.agent_id == "reachy"
    assert [card.model_dump() for card in private_states["reachy"].hole_cards] == [
        {"rank": "9", "suit": "clubs"},
        {"rank": "9", "suit": "diamonds"},
    ]
    assert "hole_cards" not in str(public_payload)
    assert "agent_decision" not in {event.event_type for event in result.events}


def test_orchestrator_rejects_invalid_agent_decision_and_keeps_agent_paused():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()
    orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "call"}}))
    _record_private(orchestrator, "reachy")

    result = orchestrator.submit_agent_decision(_decision("reachy", "bet", 100))

    assert result.accepted is False
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.type == PendingInputType.AGENT_ACTION
    assert result.state.waiting_for.agent_id == "reachy"
    assert "agent_decision_failed" in {event.event_type for event in result.events}


def test_orchestrator_agent_fallback_speech_uses_contemplation_break():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()
    orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "call"}}))
    _record_private(orchestrator, "reachy")

    result = orchestrator.submit_agent_decision(
        AgentDecision(
            agent_id="reachy",
            action=PokerAction.model_validate({"type": "call"}),
            speech=None,
            reaction={"intent": "announce_action"},
            confidence=0.9,
        )
    )

    speech = [
        event.payload.get("speech")
        for event in result.events
        if event.event_type == "presentation_command" and event.payload.get("target_client") == "reachy"
    ]
    assert speech == ['That price is workable. <break time="0.8s" /> Reachy call.']
    line = speech[0]
    assert isinstance(line, str)
    assert line.count('<break time="0.8s" />') == 1


def test_orchestrator_human_table_talk_keeps_human_action_pending():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()

    result = orchestrator.submit_human_table_talk(
        HumanTableTalkInput(
            target_agent_id="reachy",
            message="what do you think",
            raw_transcript="Reachy, what do you think?",
        ),
        speech="I am calibrating the vibes.",
        reaction={"intent": "table_talk_reply"},
        emotion="celebrate",
    )

    assert result.accepted is True
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.type == PendingInputType.HUMAN_ACTION
    assert result.state.pot == 30
    event_types = [event.event_type for event in result.events]
    assert event_types == [EventType.HUMAN_TABLE_TALK, EventType.PRESENTATION_COMMAND]
    assert result.events[1].payload["target_client"] == "reachy"
    assert result.events[1].payload["emotion"] == "celebrate"


def test_orchestrator_postflop_bet_runs_until_turn_recognition_after_agent_calls():
    orchestrator = InMemoryOrchestrator()
    _open_human_action_after_flop(orchestrator)

    result = orchestrator.submit_human_action(
        HumanActionInput.model_validate({"source": "voice", "action": {"type": "bet", "amount": 100}})
    )
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.agent_id == "reachy"
    orchestrator.submit_agent_decision(_decision("reachy", "call"))
    result = orchestrator.submit_agent_decision(_decision("eliza", "call"))

    assert result.accepted is True
    assert result.state.pot == 360
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.type == PendingInputType.PUBLIC_BOARD_CARDS
    assert result.state.board_recognition.expected_card_count == 4


def test_orchestrator_reveals_checked_river_showdown_in_table_order():
    orchestrator = InMemoryOrchestrator()
    _open_human_action_after_flop(orchestrator)
    _check_postflop_round(orchestrator)
    _commit_turn(orchestrator)
    _check_postflop_round(orchestrator)
    _commit_river(orchestrator)
    _check_postflop_round(orchestrator)

    state = orchestrator.public_state()
    assert state.waiting_for is not None
    assert state.waiting_for.type == PendingInputType.REVEALED_CARDS
    assert state.waiting_for.seat == 2
    assert state.showdown.reveal_order == [2, 3, 1]

    orchestrator.record_revealed_cards(2, [_card("king", "clubs"), _card("king", "diamonds")], source="seat_2_crop")
    orchestrator.record_revealed_cards(3, [_card("9", "clubs"), _card("9", "diamonds")], source="seat_3_crop")
    result = orchestrator.record_revealed_cards(
        1,
        [_card("queen", "hearts"), _card("jack", "hearts")],
        source="seat_1_crop",
    )

    assert result.accepted is True
    resolved_event = next(event for event in result.events if event.event_type == "showdown_resolved")
    assert resolved_event.payload["winner_seats"] == [2]
    assert result.state.hand_number == 2
    assert result.state.street == Street.PREFLOP
    assert result.state.waiting_for is not None


def test_orchestrator_auto_starts_next_hand_after_uncontested_pot():
    orchestrator = InMemoryOrchestrator()
    _open_human_action_after_flop(orchestrator)

    orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "bet", "amount": 100}}))
    orchestrator.submit_agent_decision(_decision("reachy", "fold"))
    result = orchestrator.submit_agent_decision(_decision("eliza", "fold"))

    assert result.accepted is True
    event_types = [event.event_type for event in result.events]
    assert event_types.index(EventType.SHOWDOWN_RESOLVED) < event_types.index(EventType.HAND_STARTED)
    assert result.state.hand_number == 2
    assert result.state.dealer_seat == 2
    assert result.state.small_blind_seat == 3
    assert result.state.big_blind_seat == 1
    assert result.state.players[1].stack == 2020
    assert result.state.pot == 30
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.agent_id == "reachy"


def test_orchestrator_rejects_mismatched_private_agent_path():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()
    orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "call"}}))
    observation = PrivateCardObservation(
        agent_id="eliza",
        seat=3,
        hole_cards=[_card("9", "clubs"), _card("9", "diamonds")],
        source="eliza_browser_webcam",
    )

    with pytest.raises(ValueError, match="does not match"):
        orchestrator.record_client_private_cards("reachy", observation)


def _complete_preflop(orchestrator: InMemoryOrchestrator) -> ExternalInputResult:
    orchestrator.start_game()
    orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "call"}}))
    _record_private(orchestrator, "reachy", [_card("king", "clubs"), _card("king", "diamonds")])
    orchestrator.submit_agent_decision(_decision("reachy", "call"))
    _record_private(orchestrator, "eliza")
    return orchestrator.submit_agent_decision(_decision("eliza", "check"))


def _open_human_action_after_flop(orchestrator: InMemoryOrchestrator) -> None:
    _complete_preflop(orchestrator)
    _record_public_observation(orchestrator, _flop())
    _record_public_observation(orchestrator, _flop())
    orchestrator.submit_agent_decision(_decision("reachy", "check"))
    orchestrator.submit_agent_decision(_decision("eliza", "check"))


def _check_postflop_round(orchestrator: InMemoryOrchestrator) -> None:
    orchestrator.submit_agent_decision(_decision("reachy", "check"))
    orchestrator.submit_agent_decision(_decision("eliza", "check"))
    orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "check"}}))


def _record_private(
    orchestrator: InMemoryOrchestrator,
    agent_id: str,
    cards: list[Card] | None = None,
) -> ExternalInputResult:
    return orchestrator.record_client_private_cards(
        agent_id,
        PrivateCardObservation(
            agent_id=agent_id,
            seat={"reachy": 2, "eliza": 3}[agent_id],
            hole_cards=cards or [_card("9", "clubs"), _card("9", "diamonds")],
            source=f"{agent_id}_camera",
            confidence=0.91,
        ),
    )


def _decision(agent_id: str, action_type: str, amount: int | None = None) -> AgentDecision:
    return AgentDecision(
        agent_id=agent_id,
        action=PokerAction.model_validate({"type": action_type, "amount": amount}),
        speech=f"{agent_id} {action_type}",
        reaction={"intent": "announce_action"},
        confidence=0.9,
    )


def _card(rank: str, suit: str) -> Card:
    return Card.model_validate({"rank": rank, "suit": suit})


def _flop() -> list[Card]:
    return [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")]


def _record_public_observation(
    orchestrator: InMemoryOrchestrator,
    cards: list[Card],
    *,
    confidence: float = 0.9,
) -> None:
    orchestrator.record_public_observation(
        PublicTableObservation(
            board_cards=cards,
            street_hint={0: Street.PREFLOP, 3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}.get(len(cards)),
            confidence=confidence,
        )
    )


def _commit_turn(orchestrator: InMemoryOrchestrator) -> ObservationReceipt:
    turn = [*_flop(), _card("king", "spades")]
    _record_public_observation(orchestrator, turn)
    return orchestrator.record_public_observation(
        PublicTableObservation(board_cards=turn, street_hint=Street.TURN, confidence=0.9)
    )


def _commit_river(orchestrator: InMemoryOrchestrator) -> ObservationReceipt:
    river = [*_flop(), _card("king", "spades"), _card("9", "spades")]
    _record_public_observation(orchestrator, river)
    return orchestrator.record_public_observation(
        PublicTableObservation(board_cards=river, street_hint=Street.RIVER, confidence=0.9)
    )
