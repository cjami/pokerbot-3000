import pytest

from pokerbot_3000.domain.cards import Card
from pokerbot_3000.domain.models import HumanActionInput, PrivateCardObservation, PublicTableObservation, Street
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


def test_orchestrator_start_game_pauses_for_first_human_action():
    orchestrator = InMemoryOrchestrator()

    result = orchestrator.start_game()

    assert result.accepted is True
    assert result.state.automation_status == "waiting_for_external_input"
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.type == "human_action"
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
    request = HumanActionInput.model_validate({"source": "voice", "action": {"type": "bet", "amount": 100}})

    result = orchestrator.submit_human_action(request)

    assert result.accepted is True
    assert result.state.pot == 100
    assert result.state.active_player_seat == 3
    assert result.state.waiting_for is not None
    assert result.state.waiting_for.agent_id == "eliza"
    assert orchestrator.events()[-1].event_type == "engine_paused"


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
