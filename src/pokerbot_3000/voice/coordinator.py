"""Runtime coordinator for human voice actions."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from pokerbot_3000.domain.models import GameEvent, PendingInputType

if TYPE_CHECKING:
    from pokerbot_3000.orchestrator import InMemoryOrchestrator
    from pokerbot_3000.ports.voice import (
        AudioChunk,
        AudioInput,
        SpeechTranscriber,
        VoiceActivityDetector,
        VoiceCommandParser,
    )

HUMAN_SEAT: Final = 1

type EventHook = Callable[[list[GameEvent]], Awaitable[None]]
type SnapshotPublisher = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class VoiceActionAdapters:
    """Voice adapters used by the runtime coordinator."""

    audio_input: AudioInput
    vad: VoiceActivityDetector
    transcriber: SpeechTranscriber
    parser: VoiceCommandParser


@dataclass(slots=True)
class VoiceInputStatus:
    """Public status for the server-side voice worker."""

    state: str = "starting"
    latest_transcript: str | None = None
    latest_action: dict[str, Any] | None = None
    last_rejection: str | None = None
    last_error: str | None = None

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-ready status shape."""
        return {
            "state": self.state,
            "latest_transcript": self.latest_transcript,
            "latest_action": self.latest_action,
            "last_rejection": self.last_rejection,
            "last_error": self.last_error,
        }


class VoiceActionCoordinator:
    """Listen for human speech and submit parsed poker actions."""

    def __init__(
        self,
        *,
        orchestrator: InMemoryOrchestrator,
        adapters: VoiceActionAdapters,
        after_events: EventHook | None = None,
        publish_snapshot: SnapshotPublisher | None = None,
        human_seat: int = HUMAN_SEAT,
    ) -> None:
        """Create a coordinator from explicit adapters."""
        self._orchestrator = orchestrator
        self._adapters = adapters
        self._after_events = after_events
        self._publish_snapshot = publish_snapshot
        self._human_seat = human_seat
        self._status = VoiceInputStatus()
        self._task: asyncio.Task[None] | None = None

    @property
    def status(self) -> VoiceInputStatus:
        """Return the latest public voice-worker status."""
        return self._status

    def start(self) -> None:
        """Start the background voice worker."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="pokerbot-voice-actions")

    async def stop(self) -> None:
        """Stop the background voice worker."""
        if self._task is None:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None
        self._status.state = "stopped"
        await self._publish()

    async def _run(self) -> None:
        self._status.state = "waiting_for_turn"
        await self._publish()
        try:
            segments = self._adapters.vad.speech_segments(self._adapters.audio_input.chunks())
            async for segment in segments:
                if not self._is_waiting_for_human():
                    self._status.state = "waiting_for_turn"
                    await self._publish()
                    continue
                await self._process_segment(segment)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._status.state = "error"
            self._status.last_error = str(exc) or exc.__class__.__name__
            await self._publish()

    async def _process_segment(self, segment: AudioChunk) -> None:
        self._status.state = "transcribing"
        self._status.last_error = None
        self._status.last_rejection = None
        await self._publish()

        transcript = await self._adapters.transcriber.transcribe(segment)
        self._status.latest_transcript = transcript.text
        request = self._adapters.parser.parse(transcript)
        if request is None:
            self._status.state = "listening"
            self._status.latest_action = None
            self._status.last_rejection = "Transcript did not contain a clear poker action."
            await self._publish()
            return

        self._status.state = "submitting"
        self._status.latest_action = request.action.model_dump(mode="json")
        await self._publish()

        event_start = self._orchestrator.event_count()
        try:
            result = self._orchestrator.submit_human_action(request)
        except Exception as exc:  # noqa: BLE001
            self._status.state = "error"
            self._status.last_error = str(exc) or exc.__class__.__name__
            await self._publish()
            return

        events = self._orchestrator.events_since(event_start)
        if self._after_events is not None:
            await self._after_events(events)
        if not result.accepted:
            self._status.state = "listening"
            self._status.last_rejection = result.reason
        else:
            self._status.state = "waiting_for_turn"
        await self._publish()

    def _is_waiting_for_human(self) -> bool:
        state = self._orchestrator.public_state()
        return (
            state.waiting_for is not None
            and state.waiting_for.type == PendingInputType.HUMAN_ACTION
            and state.waiting_for.seat == self._human_seat
        )

    async def _publish(self) -> None:
        if self._publish_snapshot is not None:
            await self._publish_snapshot()
