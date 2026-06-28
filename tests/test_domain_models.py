import pytest
from pydantic import TypeAdapter, ValidationError

from pokerbot_3000.domain.cards import Card
from pokerbot_3000.domain.models import (
    ActionType,
    HumanActionInput,
    PokerAction,
    PrivateCardObservation,
    PublicGameState,
    Street,
)
from pokerbot_3000.orchestrator.service import DemoDefaults, InMemoryOrchestrator


def test_card_notation_accepts_compact_rank_and_suit():
    adapter = TypeAdapter(Card)

    assert adapter.validate_python("As") == "As"
    assert adapter.validate_python("Td") == "Td"


def test_card_notation_rejects_invalid_values():
    adapter = TypeAdapter(Card)

    with pytest.raises(ValidationError):
        adapter.validate_python("10h")


def test_amount_actions_require_amounts():
    with pytest.raises(ValidationError):
        PokerAction(type=ActionType.BET)


def test_human_action_input_accepts_voice_shape():
    request = HumanActionInput.model_validate(
        {
            "source": "voice",
            "action": {"type": "raise_to", "amount": 200},
            "confidence": 0.9,
        }
    )

    assert request.seat == 1
    assert request.action.amount == 200


def test_public_state_rejects_impossible_board_counts():
    orchestrator = InMemoryOrchestrator(DemoDefaults())
    state = orchestrator.public_state().model_dump()
    state["board"] = ["As", "Kd"]

    with pytest.raises(ValidationError):
        PublicGameState.model_validate(state)


def test_private_observation_tracks_only_agent_cards():
    observation = PrivateCardObservation(
        agent_id="eliza",
        seat=3,
        hole_cards=["9c", "9d"],
        source="eliza_browser_webcam",
        confidence=0.89,
    )

    assert observation.agent_id == "eliza"
    assert observation.hole_cards == ["9c", "9d"]


def test_street_values_match_holdem_terms():
    assert Street.PREFLOP == "preflop"
