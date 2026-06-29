"""Browser-fed public table frames and Gemma vision source."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from dotenv import load_dotenv

from pokerbot_3000.domain.models import PublicTableObservation, Street
from pokerbot_3000.llm import CerebrasConfig, CerebrasLlmClient

if TYPE_CHECKING:
    from pathlib import Path

    from pokerbot_3000.domain.cards import Card
    from pokerbot_3000.domain.models import PrivateCardObservation
    from pokerbot_3000.ports.llm import ImageFrame, LlmGateway

BOARD_CARD_READER_CONFIDENCE: Final = 0.86


class GemmaPublicVisionSource:
    """Public vision source that asks Gemma to read a submitted browser frame."""

    def __init__(self, llm: LlmGateway) -> None:
        """Create a source from a task-level LLM gateway."""
        self._llm = llm
        self._latest_frame: ImageFrame | None = None

    @property
    def latest_frame(self) -> ImageFrame | None:
        """Return the latest frame sent to Gemma."""
        return self._latest_frame

    async def observe_frame(self, frame: ImageFrame) -> PublicTableObservation:
        """Interpret one browser-submitted public table frame."""
        self._latest_frame = frame
        board_cards = await self._llm.read_board_cards(frame)
        if board_cards:
            return PublicTableObservation(
                source=frame.source,
                board_cards=board_cards,
                street_hint=_street_hint_for_count(len(board_cards)),
                confidence=BOARD_CARD_READER_CONFIDENCE,
                notes="Read by board-card-specific Gemma prompt.",
            )
        return await self._llm.read_public_table(frame)


class LazyGemmaPublicVisionSource:
    """Lazily create the Gemma client on first browser frame."""

    def __init__(self, env_file: str | Path | None = None) -> None:
        """Delay configuration loading until the recognition loop actually runs."""
        self._env_file = env_file
        self._source: GemmaPublicVisionSource | None = None

    @property
    def latest_frame(self) -> ImageFrame | None:
        """Return the latest frame sent to Gemma."""
        return None if self._source is None else self._source.latest_frame

    async def observe_frame(self, frame: ImageFrame) -> PublicTableObservation:
        """Interpret a browser-submitted frame using lazily initialized dependencies."""
        if self._source is None:
            self._ensure_env_loaded()
            self._source = GemmaPublicVisionSource(
                llm=CerebrasLlmClient(CerebrasConfig.from_env(self._env_file)),
            )
        return await self._source.observe_frame(frame)

    def _ensure_env_loaded(self) -> None:
        load_dotenv(dotenv_path=self._env_file)


class GemmaPrivateCardSource:
    """Private-card source that asks Gemma to read a submitted client frame."""

    def __init__(self, llm: LlmGateway) -> None:
        """Create a source from a task-level LLM gateway."""
        self._llm = llm

    async def read_private_cards(self, agent_id: str, frame: ImageFrame) -> PrivateCardObservation:
        """Interpret one submitted private-card frame."""
        return await self._llm.read_hole_cards(agent_id, frame)


class LazyGemmaPrivateCardSource:
    """Lazily create the Gemma client on first private-card frame."""

    def __init__(self, env_file: str | Path | None = None) -> None:
        """Delay configuration loading until a thin client submits a frame."""
        self._env_file = env_file
        self._source: GemmaPrivateCardSource | None = None

    async def read_private_cards(self, agent_id: str, frame: ImageFrame) -> PrivateCardObservation:
        """Interpret a client-submitted private-card frame using lazily initialized dependencies."""
        if self._source is None:
            self._ensure_env_loaded()
            self._source = GemmaPrivateCardSource(
                llm=CerebrasLlmClient(CerebrasConfig.from_env(self._env_file)),
            )
        return await self._source.read_private_cards(agent_id, frame)

    def _ensure_env_loaded(self) -> None:
        load_dotenv(dotenv_path=self._env_file)


class GemmaRevealedCardsSource:
    """Showdown reveal source that asks Gemma to read a submitted seat crop."""

    def __init__(self, llm: LlmGateway) -> None:
        """Create a source from a task-level LLM gateway."""
        self._llm = llm

    async def read_revealed_cards(self, frame: ImageFrame) -> list[Card]:
        """Interpret one browser-submitted revealed-card crop."""
        return await self._llm.read_revealed_cards(frame)


class LazyGemmaRevealedCardsSource:
    """Lazily create the Gemma client on first revealed-card frame."""

    def __init__(self, env_file: str | Path | None = None) -> None:
        """Delay configuration loading until the recognition loop actually runs."""
        self._env_file = env_file
        self._source: GemmaRevealedCardsSource | None = None

    async def read_revealed_cards(self, frame: ImageFrame) -> list[Card]:
        """Interpret a browser-submitted reveal crop using lazily initialized dependencies."""
        if self._source is None:
            self._ensure_env_loaded()
            self._source = GemmaRevealedCardsSource(
                llm=CerebrasLlmClient(CerebrasConfig.from_env(self._env_file)),
            )
        return await self._source.read_revealed_cards(frame)

    def _ensure_env_loaded(self) -> None:
        load_dotenv(dotenv_path=self._env_file)


def _street_hint_for_count(card_count: int) -> Street | None:
    streets = {
        0: Street.PREFLOP,
        3: Street.FLOP,
        4: Street.TURN,
        5: Street.RIVER,
    }
    return streets.get(card_count)
