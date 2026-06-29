"""Voice command parsing protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from pokerbot_3000.domain.models import HumanActionInput, HumanTableTalkInput

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


type VoiceCommand = HumanActionInput | HumanTableTalkInput


@dataclass(frozen=True, slots=True)
class VoiceTranscript:
    """Speech recognition output before command parsing."""

    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """One chunk or segment of mono PCM audio."""

    pcm: bytes
    sample_rate: int = 16_000
    sample_width: int = 2


class VoiceCommandParser(Protocol):
    """Parser that turns transcripts into proposed actions."""

    def parse(self, transcript: VoiceTranscript) -> VoiceCommand | None:
        """Parse a transcript into a proposed action when possible."""


class AudioInput(Protocol):
    """Source of raw microphone audio chunks."""

    def chunks(self) -> AsyncIterator[AudioChunk]:
        """Yield microphone audio chunks until stopped."""


class VoiceActivityDetector(Protocol):
    """Detector that segments raw audio into speech phrases."""

    def speech_segments(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[AudioChunk]:
        """Yield full speech segments from raw audio chunks."""


class SpeechTranscriber(Protocol):
    """ASR adapter that turns speech audio into text."""

    async def transcribe(self, segment: AudioChunk) -> VoiceTranscript:
        """Transcribe one speech segment."""
