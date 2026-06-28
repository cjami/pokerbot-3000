"""Application runtime wiring for browser frames, voice, and live dashboard events."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import WebSocket, WebSocketDisconnect

from pokerbot_3000.domain.models import EventType, ExternalInputResult
from pokerbot_3000.orchestrator import InMemoryOrchestrator
from pokerbot_3000.perception import LazyGemmaPublicVisionSource
from pokerbot_3000.voice import (
    ElevenLabsClient,
    ElevenLabsClientError,
    ElevenLabsConfig,
)

if TYPE_CHECKING:
    from pokerbot_3000.domain.models import GameEvent, OperatorControlResult
    from pokerbot_3000.ports.llm import ImageFrame
    from pokerbot_3000.ports.perception import PublicVisionSource

type DashboardMessage = dict[str, Any]
type SnapshotFactory = Callable[[], DashboardMessage]
type EventHook = Callable[[list[GameEvent]], Awaitable[None]]


class SpeechSynthesisClient(Protocol):
    """Minimal speech client contract used by the app runtime."""

    async def synthesize_orchestrator(self, text: str) -> bytes:
        """Synthesize orchestrator speech."""


type VoiceClientFactory = Callable[[], SpeechSynthesisClient]


class DashboardEventBroadcaster:
    """Fan out live dashboard snapshots over WebSocket connections."""

    def __init__(self, snapshot_factory: SnapshotFactory) -> None:
        """Create a broadcaster that can materialize fresh snapshots."""
        self._snapshot_factory = snapshot_factory
        self._subscribers: set[asyncio.Queue[DashboardMessage]] = set()

    async def websocket_endpoint(self, websocket: WebSocket) -> None:
        """Serve one dashboard WebSocket connection."""
        await websocket.accept()
        queue: asyncio.Queue[DashboardMessage] = asyncio.Queue(maxsize=10)
        self._subscribers.add(queue)
        try:
            await websocket.send_json(self._snapshot_factory())
            while True:
                await websocket.send_json(await queue.get())
        except WebSocketDisconnect:
            return
        finally:
            self._subscribers.discard(queue)

    async def publish_snapshot(self) -> None:
        """Publish the latest full dashboard snapshot."""
        await self.publish(self._snapshot_factory())

    async def publish(self, message: DashboardMessage) -> None:
        """Publish a message to every connected dashboard."""
        for queue in self._subscribers:
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with suppress(asyncio.QueueFull):
                queue.put_nowait(message)


class PublicBoardFrameProcessor:
    """Process browser-submitted public-board frames."""

    def __init__(
        self,
        orchestrator: InMemoryOrchestrator,
        public_vision: PublicVisionSource,
        broadcaster: DashboardEventBroadcaster,
        *,
        after_events: EventHook | None = None,
    ) -> None:
        """Create a frame processor around a public vision source and orchestrator."""
        self._orchestrator = orchestrator
        self._public_vision = public_vision
        self._broadcaster = broadcaster
        self._after_events = after_events
        self._lock = asyncio.Lock()
        self._latest_frame: ImageFrame | None = None

    @property
    def is_waiting_for_frames(self) -> bool:
        """Return whether submitted frames can currently advance board recognition."""
        return self._orchestrator.needs_public_board_observation()

    @property
    def latest_frame(self) -> ImageFrame | None:
        """Return the latest camera frame sent to Gemma."""
        return self._latest_frame

    async def process_frame(self, frame: ImageFrame) -> ExternalInputResult:
        """Ask Gemma to read one submitted frame and advance the orchestrator if valid."""
        async with self._lock:
            if not self._orchestrator.needs_public_board_observation():
                return ExternalInputResult(
                    accepted=False,
                    reason="The engine is not waiting for public board cards.",
                    events=[],
                    state=self._orchestrator.public_state(),
                )

            event_start = self._orchestrator.event_count()
            self._latest_frame = frame
            try:
                observation = await self._public_vision.observe_frame(frame)
            except Exception as exc:  # noqa: BLE001
                self._orchestrator.record_public_board_error(_public_error_message(exc), source=frame.source)
                reason = _public_error_message(exc)
                accepted = False
            else:
                self._orchestrator.record_public_observation(observation)
                reason = "Processed browser public-board frame."
                accepted = True

            events = self._orchestrator.events_since(event_start)
            if self._after_events is not None:
                await self._after_events(events)
            await self._broadcaster.publish_snapshot()
            return ExternalInputResult(
                accepted=accepted,
                reason=reason,
                events=events,
                state=self._orchestrator.public_state(),
            )


def _default_voice_client_factory() -> ElevenLabsClient:
    return ElevenLabsClient(ElevenLabsConfig.from_env())


@dataclass(slots=True)
class DashboardRuntime:
    """Top-level app runtime shared by API routes and WebSockets."""

    orchestrator: InMemoryOrchestrator
    broadcaster: DashboardEventBroadcaster
    board_processor: PublicBoardFrameProcessor
    voice_client_factory: VoiceClientFactory = _default_voice_client_factory
    _voice_client: SpeechSynthesisClient | None = field(default=None, init=False)
    _audio_cache: dict[str, bytes] = field(default_factory=dict, init=False)
    _audio_tasks: dict[str, asyncio.Task[bytes]] = field(default_factory=dict, init=False)

    @classmethod
    def create_default(cls) -> DashboardRuntime:
        """Create the default production runtime."""
        orchestrator = InMemoryOrchestrator()
        runtime_ref: dict[str, DashboardRuntime] = {}

        def snapshot_factory() -> DashboardMessage:
            return runtime_ref["runtime"].snapshot()

        async def handle_events(events: list[GameEvent]) -> None:
            await runtime_ref["runtime"].handle_new_events(events)

        broadcaster = DashboardEventBroadcaster(snapshot_factory)
        board_processor = PublicBoardFrameProcessor(
            orchestrator=orchestrator,
            public_vision=LazyGemmaPublicVisionSource(),
            broadcaster=broadcaster,
            after_events=handle_events,
        )
        runtime = cls(orchestrator=orchestrator, broadcaster=broadcaster, board_processor=board_processor)
        runtime_ref["runtime"] = runtime
        return runtime

    async def start_game(self) -> OperatorControlResult:
        """Start the game and wait for browser-submitted recognition frames."""
        result = self.orchestrator.start_game()
        if result.accepted:
            await self.handle_new_events(result.events)
        await self.broadcaster.publish_snapshot()
        return result

    async def stop_game(self) -> OperatorControlResult:
        """Stop the game and clear pending recognition work."""
        result = self.orchestrator.stop_game()
        await self.broadcaster.publish_snapshot()
        return result

    async def shutdown(self) -> None:
        """Stop runtime-owned background tasks."""
        await asyncio.gather(*self._audio_tasks.values(), return_exceptions=True)

    def snapshot(self) -> DashboardMessage:
        """Return a full dashboard snapshot suitable for JSON serialization."""
        return {
            "type": "snapshot",
            "state": self.orchestrator.public_state().model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in self.orchestrator.events(limit=25)],
            "private_states": [
                state.model_dump(mode="json") for state in self.orchestrator.private_states().values()
            ],
            "client_statuses": [
                status.model_dump(mode="json") for status in self.orchestrator.client_statuses().values()
            ],
        }

    def latest_public_frame(self) -> ImageFrame | None:
        """Return the latest browser public-table frame sent to Gemma."""
        return self.board_processor.latest_frame

    async def process_public_board_frame(self, frame: ImageFrame) -> ExternalInputResult:
        """Process one browser-submitted public-board frame."""
        return await self.board_processor.process_frame(frame)

    async def handle_new_events(self, events: list[GameEvent]) -> None:
        """Handle side effects for newly appended orchestrator events."""
        _ = events

    async def synthesize_orchestrator_event(self, event_id: str) -> bytes:
        """Return MPEG audio for one queued orchestrator speech event."""
        if event_id in self._audio_cache:
            return self._audio_cache[event_id]
        if event_id in self._audio_tasks:
            return await self._audio_tasks[event_id]

        task = asyncio.create_task(self._synthesize_orchestrator_event(event_id))
        self._audio_tasks[event_id] = task
        try:
            return await task
        finally:
            self._audio_tasks.pop(event_id, None)

    async def _synthesize_orchestrator_event(self, event_id: str) -> bytes:
        """Synthesize and cache one orchestrator speech event."""
        if event_id in self._audio_cache:
            return self._audio_cache[event_id]

        event = self.orchestrator.event_by_id(event_id)
        if event is None or event.event_type != EventType.PRESENTATION_COMMAND:
            msg = "Unknown orchestrator speech event."
            raise ElevenLabsClientError(msg)

        speech = event.payload.get("speech")
        voice = event.payload.get("voice")
        if not isinstance(speech, str) or voice != "orchestrator":
            msg = "Event does not contain orchestrator speech."
            raise ElevenLabsClientError(msg)

        if self._voice_client is None:
            self._voice_client = self.voice_client_factory()
        audio = await self._voice_client.synthesize_orchestrator(speech)
        self._audio_cache[event_id] = audio
        return audio


def _public_error_message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__
