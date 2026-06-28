"""Client bridge protocols for Reachy and Eliza."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pokerbot_3000.domain.models import ClientStatus, PrivateCardObservation


class PrivateCardClient(Protocol):
    """Client capable of providing private card observations."""

    async def request_private_cards(self, agent_id: str) -> PrivateCardObservation:
        """Return the latest private card observation for an agent."""


class PresentationClient(Protocol):
    """Client capable of performing speech or embodiment commands."""

    async def send_presentation_command(self, target_client: str, command: dict[str, object]) -> ClientStatus:
        """Send a symbolic presentation command to a client."""
