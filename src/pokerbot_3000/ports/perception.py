"""Perception source protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pokerbot_3000.domain.cards import Card
    from pokerbot_3000.domain.models import PrivateCardObservation, PublicTableObservation
    from pokerbot_3000.ports.llm import ImageFrame


class PublicVisionSource(Protocol):
    """Source of public table observations."""

    async def observe_frame(self, frame: ImageFrame) -> PublicTableObservation:
        """Interpret a public table frame."""


class PrivateCardSource(Protocol):
    """Source of private card observations."""

    async def read_private_cards(self, agent_id: str, frame: ImageFrame) -> PrivateCardObservation:
        """Interpret one private-card frame for one agent."""


class RevealedCardsSource(Protocol):
    """Source of public showdown reveal observations."""

    async def read_revealed_cards(self, frame: ImageFrame) -> list[Card]:
        """Interpret one revealed seat crop."""
