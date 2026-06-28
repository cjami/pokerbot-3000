"""Voice command parsing protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pokerbot_3000.domain.models import HumanActionInput


@dataclass(frozen=True, slots=True)
class VoiceTranscript:
    """Speech recognition output before command parsing."""

    text: str
    confidence: float


class VoiceCommandParser(Protocol):
    """Parser that turns transcripts into proposed actions."""

    def parse(self, transcript: VoiceTranscript) -> HumanActionInput | None:
        """Parse a transcript into a proposed action when possible."""
