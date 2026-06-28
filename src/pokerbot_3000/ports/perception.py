"""Perception source protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pokerbot_3000.domain.models import PrivateCardObservation, PublicTableObservation


class PublicVisionSource(Protocol):
    """Source of public table observations."""

    async def observe_public_table(self) -> PublicTableObservation:
        """Capture and interpret the public table."""


class PrivateCardSource(Protocol):
    """Source of private card observations."""

    async def observe_private_cards(self, agent_id: str) -> PrivateCardObservation:
        """Capture and interpret private cards for one agent."""
