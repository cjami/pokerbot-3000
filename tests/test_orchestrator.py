import pytest

from pokerbot_3000.domain.cards import Card
from pokerbot_3000.domain.models import (
    HumanActionInput,
    PendingInputType,
    PrivateCardObservation,
    PublicTableObservation,
    Street,
)
from pokerbot_3000.orchestrator import InMemoryOrchestrator


def test_orchestrator_initializes_three_player_demo_state():
    orchestrator = InMemoryOrchestrator()

    state = orchestrator.public_state()

    assert state.hand_id == "hand_001"
    assert len(state.players) == 3
    assert state.players[1].name == "Che"
    assert state.pot == 0
    assert state.automation_status == "stopped"
    assert state.waiting_for is None
    assert orchestrator.events()[0].event_type == "system"


def test_orchestrator_start_game_pauses_for_flop_recognition():
    orchestrator = InMemoryOrchestrator()

    result = orchestrator.start_game()

    assert result.accepted is True
    assert result.state.automation_status == "waiting_for_external_input"
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.type == PendingInputType.PUBLIC_BOARD_CARDS
    assert result.state.board_recognition.expected_card_count == 3
    assert "game_started" in {event.event_type for event in result.events}


def test_orchestrator_stop_game_clears_pending_input():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()

    result = orchestrator.stop_game()

    assert result.accepted is True
    assert result.state.automation_status == "stopped"
    assert result.state.waiting_for is None
    assert result.state.legal_actions == []


def test_orchestrator_human_action_runs_until_agent_private_cards_needed():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()
    _complete_board(orchestrator)
    request = HumanActionInput.model_validate({"source": "voice", "action": {"type": "bet", "amount": 100}})

    result = orchestrator.submit_human_action(request)

    assert result.accepted is True
    assert result.state.pot == 100
    assert result.state.active_player_seat == 3
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.agent_id == "eliza"
    assert orchestrator.events()[-1].event_type == "engine_paused"


def test_orchestrator_commits_stable_flop_turn_and_river_before_human_action():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()

    _record_public_observation(
        orchestrator,
        [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")],
    )

    state = orchestrator.public_state()
    assert state.board == []
    assert state.board_recognition.stable_sample_count == 1
    assert state.waiting_for is not None
    assert state.waiting_for.type == PendingInputType.PUBLIC_BOARD_CARDS

    _record_public_observation(
        orchestrator,
        [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")],
    )

    state = orchestrator.public_state()
    assert state.street == Street.FLOP
    assert state.board_recognition.expected_card_count == 4

    _record_public_observation(
        orchestrator,
        [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs"), _card("king", "spades")],
    )
    _record_public_observation(
        orchestrator,
        [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs"), _card("king", "spades")],
    )
    _record_public_observation(
        orchestrator,
        [
            _card("ace", "hearts"),
            _card("7", "diamonds"),
            _card("2", "clubs"),
            _card("king", "spades"),
            _card("9", "clubs"),
        ],
    )
    _record_public_observation(
        orchestrator,
        [
            _card("ace", "hearts"),
            _card("7", "diamonds"),
            _card("2", "clubs"),
            _card("king", "spades"),
            _card("9", "clubs"),
        ],
    )

    state = orchestrator.public_state()
    assert state.street == Street.RIVER
    assert state.board_recognition.status == "complete"
    assert state.waiting_for is not None
    assert state.waiting_for.type == PendingInputType.HUMAN_ACTION


def test_orchestrator_rejects_unstable_or_invalid_board_observations():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()

    _record_public_observation(
        orchestrator,
        [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")],
        confidence=0.3,
    )
    last_error = orchestrator.public_state().board_recognition.last_error
    assert last_error is not None
    assert last_error.startswith("Confidence")

    _record_public_observation(
        orchestrator,
        [_card("ace", "hearts"), _card("ace", "hearts"), _card("2", "clubs")],
    )
    assert orchestrator.public_state().board_recognition.last_error == "Detected duplicate board cards."

    _record_public_observation(
        orchestrator,
        [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")],
    )
    _record_public_observation(
        orchestrator,
        [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")],
    )
    _record_public_observation(
        orchestrator,
        [_card("ace", "spades"), _card("7", "diamonds"), _card("2", "clubs"), _card("king", "spades")],
    )

    assert orchestrator.public_state().board_recognition.last_error == "Previously committed board cards changed."


def test_orchestrator_records_public_observation_as_event_only():
    orchestrator = InMemoryOrchestrator()
    observation = PublicTableObservation(
        dealer_seat=1,
        board_cards=[
            _card("ace", "hearts"),
            _card("7", "diamonds"),
            _card("2", "clubs"),
        ],
        street_hint=Street.FLOP,
        pot_has_chips=True,
        confidence=0.84,
    )

    receipt = orchestrator.record_public_observation(observation)

    assert receipt.accepted is True
    assert receipt.event.event_type == "vision_observation"
    assert orchestrator.public_state().board == []


def test_orchestrator_runs_agent_internally_after_thin_client_cards():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()
    _complete_board(orchestrator)
    orchestrator.submit_human_action(
        HumanActionInput.model_validate({"source": "voice", "action": {"type": "bet", "amount": 100}})
    )
    observation = PrivateCardObservation(
        agent_id="eliza",
        seat=3,
        hole_cards=[
            _card("9", "clubs"),
            _card("9", "diamonds"),
        ],
        source="eliza_browser_webcam",
        confidence=0.91,
    )

    result = orchestrator.record_client_private_cards("eliza", observation)

    private_states = orchestrator.private_states()
    public_payload = orchestrator.public_state().model_dump()
    assert result.accepted is True
    assert result.state.pot == 200
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.agent_id == "reachy"
    assert [card.model_dump() for card in private_states["eliza"].hole_cards] == [
        {"rank": "9", "suit": "clubs"},
        {"rank": "9", "suit": "diamonds"},
    ]
    assert "hole_cards" not in str(public_payload)
    assert "agent_decision" in {event.event_type for event in result.events}


def test_orchestrator_rejects_mismatched_private_agent_path():
    orchestrator = InMemoryOrchestrator()
    orchestrator.start_game()
    _complete_board(orchestrator)
    orchestrator.submit_human_action(
        HumanActionInput.model_validate({"source": "voice", "action": {"type": "bet", "amount": 100}})
    )
    observation = PrivateCardObservation(
        agent_id="eliza",
        seat=3,
        hole_cards=[
            _card("9", "clubs"),
            _card("9", "diamonds"),
        ],
        source="eliza_browser_webcam",
    )

    with pytest.raises(ValueError, match="does not match"):
        orchestrator.record_client_private_cards("reachy", observation)


def _card(rank: str, suit: str) -> Card:
    return Card.model_validate({"rank": rank, "suit": suit})


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


def _complete_board(orchestrator: InMemoryOrchestrator) -> None:
    flop = [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")]
    turn = [*flop, _card("king", "spades")]
    river = [*turn, _card("9", "clubs")]
    for cards in (flop, flop, turn, turn, river, river):
        _record_public_observation(orchestrator, cards)
