import asyncio
from typing import cast

from pokerbot_3000.domain.cards import Card
from pokerbot_3000.domain.models import PublicTableObservation
from pokerbot_3000.perception.public_vision import BOARD_CARD_READER_CONFIDENCE, GemmaPublicVisionSource
from pokerbot_3000.ports.llm import ImageFrame, LlmGateway


class FakeLlm:
    """LLM fake for public vision tests."""

    def __init__(self, board_cards: list[Card]) -> None:
        """Configure board-card results for the fake."""
        self.board_cards = board_cards
        self.public_table_calls = 0

    async def read_board_cards(self, _frame: ImageFrame) -> list[Card]:
        """Return configured board-card results."""
        return self.board_cards

    async def read_public_table(self, _frame: ImageFrame) -> PublicTableObservation:
        """Return a fallback broad public-table observation."""
        self.public_table_calls += 1
        return PublicTableObservation(source="fallback", confidence=0.2)


def test_public_vision_reads_board_cards_before_broad_table_state():
    cards = [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")]
    llm = FakeLlm(cards)
    source = GemmaPublicVisionSource(llm=cast("LlmGateway", llm))

    observation = asyncio.run(source.observe_frame(_frame()))

    assert observation.source == "test_camera"
    assert observation.board_cards == cards
    assert observation.confidence == BOARD_CARD_READER_CONFIDENCE
    assert observation.street_hint == "flop"
    assert source.latest_frame is not None
    assert llm.public_table_calls == 0


def test_public_vision_falls_back_to_broad_table_state_when_no_cards_are_read():
    llm = FakeLlm([])
    source = GemmaPublicVisionSource(llm=cast("LlmGateway", llm))

    observation = asyncio.run(source.observe_frame(_frame()))

    assert observation.source == "fallback"
    assert llm.public_table_calls == 1


def _card(rank: str, suit: str) -> Card:
    return Card.model_validate({"rank": rank, "suit": suit})


def _frame() -> ImageFrame:
    return ImageFrame(source="test_camera", data_uri="data:image/jpeg;base64,dGVzdA==")
