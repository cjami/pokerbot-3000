import pytest
from pydantic import TypeAdapter, ValidationError

from pokerbot_3000.domain.cards import Card, CardRank, CardSuit
from pokerbot_3000.domain.models import (
    ActionType,
    HumanActionInput,
    PokerAction,
    PrivateCardObservation,
    PublicGameState,
    Street,
)
from pokerbot_3000.orchestrator import DemoDefaults, InMemoryOrchestrator


def test_card_accepts_structured_rank_and_suit():
    adapter = TypeAdapter(Card)

    card = adapter.validate_python({"rank": "ace", "suit": "spades"})

    assert card.rank == CardRank.ACE
    assert card.suit == CardSuit.SPADES
    assert card.label == "Ace of Spades"


def test_card_rejects_unknown_rank_and_suit_values():
    adapter = TypeAdapter(Card)

    with pytest.raises(ValidationError):
        adapter.validate_python({"rank": "Ace", "suit": "spade"})


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
    state["board"] = [
        {"rank": "ace", "suit": "spades"},
        {"rank": "king", "suit": "diamonds"},
    ]

    with pytest.raises(ValidationError):
        PublicGameState.model_validate(state)


def test_private_observation_tracks_only_agent_cards():
    observation = PrivateCardObservation(
        agent_id="eliza",
        seat=3,
        hole_cards=[
            _card("9", "clubs"),
            _card("9", "diamonds"),
        ],
        source="eliza_browser_webcam",
        confidence=0.89,
    )

    assert observation.agent_id == "eliza"
    assert observation.hole_cards[0].rank == "9"
    assert observation.hole_cards[0].suit == "clubs"


def test_street_values_match_holdem_terms():
    assert Street.PREFLOP == "preflop"


def _card(rank: str, suit: str) -> Card:
    return Card.model_validate({"rank": rank, "suit": suit})
