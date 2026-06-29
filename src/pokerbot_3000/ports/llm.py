"""LLM gateway protocol for Cerebras-hosted Gemma calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pokerbot_3000.domain.cards import Card
    from pokerbot_3000.domain.models import (
        GameEvent,
        HumanTableTalkInput,
        PokerAction,
        PrivateAgentState,
        PrivateCardObservation,
        PublicGameState,
        PublicTableObservation,
    )


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """Base64 or data-URI image payload captured by a client."""

    source: str
    data_uri: str
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class AgentDecision:
    """Structured agent decision returned by the LLM gateway."""

    agent_id: str
    action: PokerAction
    speech: str | None
    reaction: dict[str, object]
    confidence: float


@dataclass(frozen=True, slots=True)
class AgentBanterDecision:
    """Structured optional table-talk response returned by the LLM gateway."""

    agent_id: str | None
    speech: str | None
    reaction: dict[str, object]
    confidence: float
    emotion: str = "calm"


class LlmGateway(Protocol):
    """Task-level LLM interface owned by the Python orchestrator."""

    async def read_public_table(self, frame: ImageFrame) -> PublicTableObservation:
        """Read labelled public table state from a frame."""

    async def read_board_cards(self, frame_or_crop: ImageFrame) -> list[Card]:
        """Read visible cards from the public board zone."""

    async def read_hole_cards(self, agent_id: str, frame: ImageFrame) -> PrivateCardObservation:
        """Read private hole cards for one agent."""

    async def read_revealed_cards(self, frame: ImageFrame) -> list[Card]:
        """Read two revealed hole cards from one seat crop."""

    async def decide_agent_action(
        self,
        agent_id: str,
        public_state: PublicGameState,
        private_state: PrivateAgentState,
    ) -> AgentDecision:
        """Choose an agent action from public and private state."""

    async def generate_table_talk(self, agent_id: str, event: GameEvent) -> str:
        """Generate a table-talk line for an agent."""

    async def respond_to_human_table_talk(
        self,
        request: HumanTableTalkInput,
        public_state: PublicGameState,
    ) -> AgentBanterDecision:
        """Respond to direct human speech addressed to one agent."""

    async def react_to_human_action(
        self,
        event: GameEvent,
        public_state: PublicGameState,
    ) -> AgentBanterDecision:
        """Optionally react to one committed human poker action."""
