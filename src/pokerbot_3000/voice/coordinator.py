"""Runtime coordinator for human voice actions."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from pokerbot_3000.domain.models import (
    ActionType,
    ExternalInputResult,
    GameEvent,
    HumanActionInput,
    HumanTableTalkInput,
    PendingInputType,
    PokerAction,
)

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
LOGGER = logging.getLogger(__name__)

type EventHook = Callable[[list[GameEvent]], Awaitable[None]]
type SnapshotPublisher = Callable[[], Awaitable[None]]
type TableTalkSubmitter = Callable[[HumanTableTalkInput], Awaitable[ExternalInputResult]]


@runtime_checkable
class _WarmableTranscriber(Protocol):
    async def warm_up(self) -> None:
        """Prepare the transcriber for the first live phrase."""


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
    latest_table_talk: dict[str, Any] | None = None
    last_rejection: str | None = None
    last_error: str | None = None
    speech_segment_count: int = 0
    transcribed_segment_count: int = 0
    ignored_segment_count: int = 0
    transcriber_ready: bool = False

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-ready status shape."""
        return {
            "state": self.state,
            "latest_transcript": self.latest_transcript,
            "latest_action": self.latest_action,
            "latest_table_talk": self.latest_table_talk,
            "last_rejection": self.last_rejection,
            "last_error": self.last_error,
            "speech_segment_count": self.speech_segment_count,
            "transcribed_segment_count": self.transcribed_segment_count,
            "ignored_segment_count": self.ignored_segment_count,
            "transcriber_ready": self.transcriber_ready,
        }


class VoiceActionCoordinator:
    """Listen for human speech and submit parsed poker actions."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        orchestrator: InMemoryOrchestrator,
        adapters: VoiceActionAdapters,
        submit_table_talk: TableTalkSubmitter | None = None,
        after_events: EventHook | None = None,
        publish_snapshot: SnapshotPublisher | None = None,
        human_seat: int = HUMAN_SEAT,
    ) -> None:
        """Create a coordinator from explicit adapters."""
        self._orchestrator = orchestrator
        self._adapters = adapters
        self._submit_table_talk = submit_table_talk
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
        try:
            await self._warm_up_transcriber()
            self._status.state = "waiting_for_turn"
            await self._publish()
            LOGGER.info("Voice input is ready.")
            segments = self._adapters.vad.speech_segments(self._adapters.audio_input.chunks())
            async for segment in segments:
                self._status.speech_segment_count += 1
                LOGGER.info(
                    "Received voice segment %d (%d bytes).",
                    self._status.speech_segment_count,
                    len(segment.pcm),
                )
                if (reason := self._not_waiting_for_human_reason()) is not None:
                    self._status.state = "waiting_for_turn"
                    self._status.ignored_segment_count += 1
                    self._status.last_rejection = f"Ignored speech because {reason}."
                    LOGGER.info("Ignored voice segment because %s.", reason)
                    await self._publish()
                    continue
                await self._process_segment(segment)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("Voice input worker failed.")
            self._status.state = "error"
            self._status.last_error = str(exc) or exc.__class__.__name__
            await self._publish()

    async def _warm_up_transcriber(self) -> None:
        transcriber = self._adapters.transcriber
        if not isinstance(transcriber, _WarmableTranscriber):
            self._status.transcriber_ready = True
            return

        self._status.state = "warming_up"
        self._status.last_error = None
        await self._publish()
        LOGGER.info("Warming up voice transcriber.")
        await transcriber.warm_up()
        self._status.transcriber_ready = True
        LOGGER.info("Voice transcriber is ready.")

    async def _process_segment(self, segment: AudioChunk) -> None:
        self._status.state = "transcribing"
        self._status.last_error = None
        self._status.last_rejection = None
        await self._publish()

        LOGGER.info("Transcribing voice segment (%d bytes).", len(segment.pcm))
        transcript = await self._adapters.transcriber.transcribe(segment)
        self._status.transcribed_segment_count += 1
        self._status.latest_transcript = transcript.text
        LOGGER.info("Voice transcript: %r.", transcript.text)
        request = self._adapters.parser.parse(transcript)
        if request is None:
            self._status.state = "listening"
            self._status.latest_action = None
            self._status.latest_table_talk = None
            self._status.last_rejection = "Transcript did not contain a clear poker action."
            LOGGER.info("Rejected voice transcript because it did not contain a clear poker action.")
            await self._publish()
            return

        if isinstance(request, HumanTableTalkInput):
            await self._submit_table_talk_request(request)
            return

        request = self._action_for_current_state(request)
        self._status.state = "submitting"
        self._status.latest_action = request.action.model_dump(mode="json")
        self._status.latest_table_talk = None
        LOGGER.info("Submitting voice action: %s.", self._status.latest_action)
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
            LOGGER.info("Voice action was rejected: %s.", result.reason)
        else:
            self._status.state = "waiting_for_turn"
            LOGGER.info("Voice action was accepted.")
        await self._publish()

    async def _submit_table_talk_request(self, request: HumanTableTalkInput) -> None:
        self._status.state = "submitting"
        self._status.latest_action = None
        self._status.latest_table_talk = request.model_dump(mode="json")
        await self._publish()

        if self._submit_table_talk is None:
            self._status.state = "listening"
            self._status.last_rejection = "Human table talk is not configured."
            await self._publish()
            return

        try:
            result = await self._submit_table_talk(request)
        except Exception as exc:  # noqa: BLE001
            self._status.state = "error"
            self._status.last_error = str(exc) or exc.__class__.__name__
            await self._publish()
            return

        if self._after_events is not None:
            await self._after_events(result.events)
        if not result.accepted:
            self._status.state = "listening"
            self._status.last_rejection = result.reason
        else:
            self._status.state = "listening"
        await self._publish()

    def _not_waiting_for_human_reason(self) -> str | None:
        state = self._orchestrator.public_state()
        waiting_for = state.waiting_for
        if waiting_for is None:
            return "no human action is pending"
        if waiting_for.type != PendingInputType.HUMAN_ACTION:
            return f"the engine is waiting for {waiting_for.type}"
        if waiting_for.seat != self._human_seat:
            return f"the engine is waiting for seat {waiting_for.seat}"
        return None

    def _action_for_current_state(self, request: HumanActionInput) -> HumanActionInput:
        state = self._orchestrator.public_state()
        if (
            request.action.type == ActionType.BET
            and ActionType.BET not in state.legal_actions
            and ActionType.RAISE_TO in state.legal_actions
        ):
            return request.model_copy(
                update={"action": PokerAction(type=ActionType.RAISE_TO, amount=request.action.amount)}
            )
        if (
            request.action.type == ActionType.RAISE_TO
            and ActionType.RAISE_TO not in state.legal_actions
            and ActionType.BET in state.legal_actions
        ):
            return request.model_copy(update={"action": PokerAction(type=ActionType.BET, amount=request.action.amount)})
        return request

    async def _publish(self) -> None:
        if self._publish_snapshot is not None:
            await self._publish_snapshot()
