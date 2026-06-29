"""Application runtime wiring for browser frames, voice, and live dashboard events."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import WebSocket, WebSocketDisconnect

from pokerbot_3000.domain.models import ClientId, EventType, ExternalInputResult
from pokerbot_3000.llm import CerebrasConfig, CerebrasLlmClient
from pokerbot_3000.orchestrator import InMemoryOrchestrator
from pokerbot_3000.perception import (
    LazyGemmaPrivateCardSource,
    LazyGemmaPublicVisionSource,
    LazyGemmaRevealedCardsSource,
)
from pokerbot_3000.voice import (
    BrowserAudioInput,
    DeterministicVoiceCommandParser,
    ElevenLabsClient,
    ElevenLabsClientError,
    ElevenLabsConfig,
    ParakeetSpeechTranscriber,
    SileroVoiceActivityDetector,
    VoiceActionAdapters,
    VoiceActionCoordinator,
    VoiceInputStatus,
)

if TYPE_CHECKING:
    from pokerbot_3000.domain.models import (
        GameEvent,
        HumanActionInput,
        HumanTableTalkInput,
        OperatorControlResult,
        PrivateAgentState,
        PrivateCardObservation,
        PublicGameState,
    )
    from pokerbot_3000.ports.llm import AgentBanterDecision, AgentDecision, ImageFrame
    from pokerbot_3000.ports.perception import PrivateCardSource, PublicVisionSource, RevealedCardsSource
    from pokerbot_3000.voice.coordinator import VoiceActionCoordinator as VoiceActionCoordinatorType

type DashboardMessage = dict[str, Any]
type SnapshotFactory = Callable[[], DashboardMessage]
type EventHook = Callable[[list[GameEvent]], Awaitable[None]]


class SpeechSynthesisClient(Protocol):
    """Minimal speech client contract used by the app runtime."""

    async def synthesize_orchestrator(self, text: str) -> bytes:
        """Synthesize orchestrator speech."""

    async def synthesize_eliza(self, text: str) -> bytes:
        """Synthesize Eliza speech."""


type VoiceClientFactory = Callable[[], SpeechSynthesisClient]


class AgentDecisionSource(Protocol):
    """Minimal LLM contract for poker agent decisions."""

    async def decide_agent_action(
        self,
        agent_id: str,
        public_state: PublicGameState,
        private_state: PrivateAgentState,
    ) -> AgentDecision:
        """Choose one poker action for the active agent."""


class AgentBanterSource(Protocol):
    """Minimal LLM contract for optional poker table talk."""

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


class LazyGemmaAgentDecisionSource:
    """Lazy Cerebras/Gemma agent decision source."""

    def __init__(self) -> None:
        """Create a lazy source without reading environment yet."""
        self._llm: CerebrasLlmClient | None = None

    async def decide_agent_action(
        self,
        agent_id: str,
        public_state: PublicGameState,
        private_state: PrivateAgentState,
    ) -> AgentDecision:
        """Choose one poker action using Gemma."""
        if self._llm is None:
            self._llm = CerebrasLlmClient(CerebrasConfig.from_env())
        return await self._llm.decide_agent_action(agent_id, public_state, private_state)


class LazyGemmaAgentBanterSource:
    """Lazy Cerebras/Gemma table-talk source."""

    def __init__(self) -> None:
        """Create a lazy source without reading environment yet."""
        self._llm: CerebrasLlmClient | None = None

    async def respond_to_human_table_talk(
        self,
        request: HumanTableTalkInput,
        public_state: PublicGameState,
    ) -> AgentBanterDecision:
        """Generate a response to human-addressed agent speech."""
        if self._llm is None:
            self._llm = CerebrasLlmClient(CerebrasConfig.from_env())
        return await self._llm.respond_to_human_table_talk(request, public_state)

    async def react_to_human_action(
        self,
        event: GameEvent,
        public_state: PublicGameState,
    ) -> AgentBanterDecision:
        """Optionally generate a reaction to one committed human action."""
        if self._llm is None:
            self._llm = CerebrasLlmClient(CerebrasConfig.from_env())
        return await self._llm.react_to_human_action(event, public_state)


class DashboardEventBroadcaster:
    """Fan out live dashboard snapshots over WebSocket connections."""

    def __init__(self, snapshot_factory: SnapshotFactory) -> None:
        """Create a broadcaster that can materialize fresh snapshots."""
        self._snapshot_factory = snapshot_factory
        self._subscribers: set[asyncio.Queue[DashboardMessage]] = set()

    async def websocket_endpoint(self, websocket: WebSocket, snapshot_factory: SnapshotFactory | None = None) -> None:
        """Serve one dashboard WebSocket connection."""
        current_snapshot = snapshot_factory or self._snapshot_factory
        await websocket.accept()
        queue: asyncio.Queue[DashboardMessage] = asyncio.Queue(maxsize=10)
        self._subscribers.add(queue)
        try:
            await websocket.send_json(current_snapshot())
            while True:
                message = await queue.get()
                await websocket.send_json(current_snapshot() if snapshot_factory is not None else message)
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


class RevealedCardsFrameProcessor:
    """Process browser-submitted showdown reveal frames."""

    def __init__(
        self,
        orchestrator: InMemoryOrchestrator,
        revealed_cards: RevealedCardsSource,
        broadcaster: DashboardEventBroadcaster,
        *,
        after_events: EventHook | None = None,
    ) -> None:
        """Create a frame processor around a revealed-card source and orchestrator."""
        self._orchestrator = orchestrator
        self._revealed_cards = revealed_cards
        self._broadcaster = broadcaster
        self._after_events = after_events
        self._lock = asyncio.Lock()

    async def process_frame(self, seat: int, frame: ImageFrame) -> ExternalInputResult:
        """Ask Gemma to read one revealed seat crop and advance the orchestrator."""
        async with self._lock:
            event_start = self._orchestrator.event_count()
            try:
                cards = await self._revealed_cards.read_revealed_cards(frame)
                result = self._orchestrator.record_revealed_cards(seat, cards, source=frame.source)
            except Exception as exc:  # noqa: BLE001
                result = self._orchestrator.record_revealed_cards(seat, [], source=frame.source)
                if result.accepted:
                    self._orchestrator.record_public_board_error(_public_error_message(exc), source=frame.source)
                else:
                    result = ExternalInputResult(
                        accepted=False,
                        reason=_public_error_message(exc),
                        events=self._orchestrator.events_since(event_start),
                        state=self._orchestrator.public_state(),
                    )

            if self._after_events is not None:
                await self._after_events(result.events)
            await self._broadcaster.publish_snapshot()
            return result


class PrivateCardsFrameProcessor:
    """Process thin-client private-card frames."""

    def __init__(
        self,
        orchestrator: InMemoryOrchestrator,
        private_cards: PrivateCardSource,
        broadcaster: DashboardEventBroadcaster,
        *,
        after_events: EventHook | None = None,
    ) -> None:
        """Create a frame processor around a private-card source and orchestrator."""
        self._orchestrator = orchestrator
        self._private_cards = private_cards
        self._broadcaster = broadcaster
        self._after_events = after_events
        self._lock = asyncio.Lock()

    async def process_frame(self, agent_id: str, frame: ImageFrame) -> ExternalInputResult:
        """Ask Gemma to read a private-card frame and advance the agent turn."""
        async with self._lock:
            event_start = self._orchestrator.event_count()
            try:
                observation = await self._private_cards.read_private_cards(agent_id, frame)
                result = self._orchestrator.record_client_private_cards(agent_id, observation)
            except Exception as exc:  # noqa: BLE001
                result = ExternalInputResult(
                    accepted=False,
                    reason=_public_error_message(exc),
                    events=self._orchestrator.events_since(event_start),
                    state=self._orchestrator.public_state(),
                )

            if self._after_events is not None:
                await self._after_events(result.events)
            await self._broadcaster.publish_snapshot()
            return result


class AgentActionProcessor:
    """Drain pending Gemma agent-action turns."""

    def __init__(
        self,
        *,
        orchestrator: InMemoryOrchestrator,
        decisions: AgentDecisionSource,
        broadcaster: DashboardEventBroadcaster,
        after_events: EventHook | None = None,
    ) -> None:
        """Create a processor around an LLM decision source."""
        self._orchestrator = orchestrator
        self._decisions = decisions
        self._broadcaster = broadcaster
        self._after_events = after_events
        self._lock = asyncio.Lock()

    async def process_pending(self) -> list[GameEvent]:
        """Process queued agent actions until the engine blocks elsewhere."""
        async with self._lock:
            all_events: list[GameEvent] = []
            while agent_id := self._orchestrator.pending_agent_action():
                event_start = self._orchestrator.event_count()
                try:
                    decision = await self._decisions.decide_agent_action(
                        agent_id,
                        self._orchestrator.public_state(),
                        self._orchestrator.private_state_for_agent(agent_id),
                    )
                    result = self._orchestrator.submit_agent_decision(decision)
                except Exception as exc:  # noqa: BLE001
                    self._orchestrator.record_agent_decision_failed(agent_id, _public_error_message(exc))
                    result = ExternalInputResult(
                        accepted=False,
                        reason=_public_error_message(exc),
                        events=self._orchestrator.events_since(event_start),
                        state=self._orchestrator.public_state(),
                    )

                all_events.extend(result.events)
                if self._after_events is not None:
                    await self._after_events(result.events)
                await self._broadcaster.publish_snapshot()
                if not result.accepted:
                    break
            return all_events


def _default_voice_client_factory() -> ElevenLabsClient:
    return ElevenLabsClient(ElevenLabsConfig.from_env())


@dataclass(slots=True)
class DashboardRuntime:
    """Top-level app runtime shared by API routes and WebSockets."""

    orchestrator: InMemoryOrchestrator
    broadcaster: DashboardEventBroadcaster
    board_processor: PublicBoardFrameProcessor
    private_cards_processor: PrivateCardsFrameProcessor
    revealed_cards_processor: RevealedCardsFrameProcessor
    agent_action_processor: AgentActionProcessor | None = None
    agent_banter_source: AgentBanterSource | None = None
    voice_coordinator: VoiceActionCoordinatorType | None = None
    browser_voice_input: BrowserAudioInput | None = None
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
        private_cards_processor = PrivateCardsFrameProcessor(
            orchestrator=orchestrator,
            private_cards=LazyGemmaPrivateCardSource(),
            broadcaster=broadcaster,
            after_events=handle_events,
        )
        revealed_cards_processor = RevealedCardsFrameProcessor(
            orchestrator=orchestrator,
            revealed_cards=LazyGemmaRevealedCardsSource(),
            broadcaster=broadcaster,
            after_events=handle_events,
        )
        agent_action_processor = AgentActionProcessor(
            orchestrator=orchestrator,
            decisions=LazyGemmaAgentDecisionSource(),
            broadcaster=broadcaster,
            after_events=handle_events,
        )
        agent_banter_source = LazyGemmaAgentBanterSource()
        browser_voice_input = BrowserAudioInput()

        async def submit_table_talk(request: HumanTableTalkInput) -> ExternalInputResult:
            return await runtime_ref["runtime"].build_human_table_talk_result(request)

        voice_coordinator = VoiceActionCoordinator(
            orchestrator=orchestrator,
            adapters=VoiceActionAdapters(
                audio_input=browser_voice_input,
                vad=SileroVoiceActivityDetector(),
                transcriber=ParakeetSpeechTranscriber(),
                parser=DeterministicVoiceCommandParser(),
            ),
            submit_table_talk=submit_table_talk,
            after_events=handle_events,
            publish_snapshot=broadcaster.publish_snapshot,
        )
        runtime = cls(
            orchestrator=orchestrator,
            broadcaster=broadcaster,
            board_processor=board_processor,
            private_cards_processor=private_cards_processor,
            revealed_cards_processor=revealed_cards_processor,
            agent_action_processor=agent_action_processor,
            agent_banter_source=agent_banter_source,
            voice_coordinator=voice_coordinator,
            browser_voice_input=browser_voice_input,
        )
        runtime_ref["runtime"] = runtime
        return runtime

    async def startup(self) -> None:
        """Start runtime-owned background workers."""
        if self.voice_coordinator is not None:
            self.voice_coordinator.start()

    async def start_game(self) -> OperatorControlResult:
        """Start the game and wait for browser-submitted recognition frames."""
        result = self.orchestrator.start_game()
        if result.accepted:
            await self.handle_new_events(result.events)
            await self.process_pending_agent_actions()
        await self.broadcaster.publish_snapshot()
        return result

    async def stop_game(self) -> OperatorControlResult:
        """Stop the game and clear pending recognition work."""
        result = self.orchestrator.stop_game()
        await self.broadcaster.publish_snapshot()
        return result

    async def shutdown(self) -> None:
        """Stop runtime-owned background tasks."""
        if self.voice_coordinator is not None:
            await self.voice_coordinator.stop()
        await asyncio.gather(*self._audio_tasks.values(), return_exceptions=True)

    def snapshot(self) -> DashboardMessage:
        """Return a full dashboard snapshot suitable for JSON serialization."""
        return {
            "type": "snapshot",
            "state": self.orchestrator.public_state().model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in self.orchestrator.events(limit=25)],
            "private_states": [state.model_dump(mode="json") for state in self.orchestrator.private_states().values()],
            "client_statuses": [
                status.model_dump(mode="json") for status in self.orchestrator.client_statuses().values()
            ],
            "voice_input": self.voice_status(),
        }

    def client_snapshot(self, client_id: ClientId) -> DashboardMessage:
        """Return a private-data-scoped snapshot for a thin client."""
        agent_id = client_id.value if client_id in {ClientId.ELIZA, ClientId.REACHY} else None
        return {
            "type": "snapshot",
            "client_id": client_id.value,
            "state": self.orchestrator.public_state().model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in self.orchestrator.events(limit=25)],
            "private_states": [
                state.model_dump(mode="json")
                for state in self.orchestrator.private_states().values()
                if state.agent_id == agent_id
            ],
            "client_statuses": [
                status.model_dump(mode="json") for status in self.orchestrator.client_statuses().values()
            ],
            "voice_input": self.voice_status(),
        }

    def voice_status(self) -> dict[str, Any]:
        """Return the public server-side voice worker status."""
        if self.voice_coordinator is None:
            return VoiceInputStatus(state="not_configured").model_dump()
        status = self.voice_coordinator.status.model_dump()
        status["audio_source"] = "browser"
        status["browser_connected"] = bool(self.browser_voice_input and self.browser_voice_input.connected)
        return status

    async def browser_voice_websocket_endpoint(self, websocket: WebSocket) -> None:
        """Receive browser microphone PCM chunks for the voice worker."""
        await websocket.accept()
        if self.browser_voice_input is None:
            await websocket.close(code=1011)
            return
        self.browser_voice_input.connect()
        await self.broadcaster.publish_snapshot()
        try:
            while True:
                await self.browser_voice_input.submit_pcm(await websocket.receive_bytes())
        except WebSocketDisconnect:
            return
        finally:
            self.browser_voice_input.disconnect()
            await self.broadcaster.publish_snapshot()

    def latest_public_frame(self) -> ImageFrame | None:
        """Return the latest browser public-table frame sent to Gemma."""
        return self.board_processor.latest_frame

    async def process_public_board_frame(self, frame: ImageFrame) -> ExternalInputResult:
        """Process one browser-submitted public-board frame."""
        result = await self.board_processor.process_frame(frame)
        return await self._with_pending_agent_events(result)

    async def process_private_cards_frame(self, agent_id: str, frame: ImageFrame) -> ExternalInputResult:
        """Process one thin-client private-card frame."""
        result = await self.private_cards_processor.process_frame(agent_id, frame)
        return await self._with_pending_agent_events(result)

    async def process_revealed_cards_frame(self, seat: int, frame: ImageFrame) -> ExternalInputResult:
        """Process one browser-submitted revealed-card frame."""
        result = await self.revealed_cards_processor.process_frame(seat, frame)
        return await self._with_pending_agent_events(result)

    async def submit_human_action(self, request: HumanActionInput) -> ExternalInputResult:
        """Consume a human action and drain any resulting agent turns."""
        result = self.orchestrator.submit_human_action(request)
        if result.accepted:
            reaction_events = await self._maybe_queue_human_action_reaction(result.events)
            if reaction_events:
                result = result.model_copy(
                    update={
                        "events": [*result.events, *reaction_events],
                        "state": self.orchestrator.public_state(),
                    },
                )
        await self.handle_new_events(result.events)
        await self.broadcaster.publish_snapshot()
        return await self._with_pending_agent_events(result)

    async def submit_human_table_talk(self, request: HumanTableTalkInput) -> ExternalInputResult:
        """Record human-addressed agent speech and publish the targeted response."""
        result = await self.build_human_table_talk_result(request)
        await self.handle_new_events(result.events)
        await self.broadcaster.publish_snapshot()
        return result

    async def record_client_private_cards(
        self,
        agent_id: str,
        observation: PrivateCardObservation,
    ) -> ExternalInputResult:
        """Consume structured private-card input and drain resulting agent turns."""
        result = self.orchestrator.record_client_private_cards(agent_id, observation)
        await self.handle_new_events(result.events)
        await self.broadcaster.publish_snapshot()
        return await self._with_pending_agent_events(result)

    async def process_pending_agent_actions(self) -> list[GameEvent]:
        """Drain pending Gemma agent decisions if configured."""
        if self.agent_action_processor is None:
            return []
        return await self.agent_action_processor.process_pending()

    async def _with_pending_agent_events(self, result: ExternalInputResult) -> ExternalInputResult:
        agent_events = await self.process_pending_agent_actions()
        if not agent_events:
            return result
        return result.model_copy(
            update={
                "events": [*result.events, *agent_events],
                "state": self.orchestrator.public_state(),
            },
        )

    async def build_human_table_talk_result(self, request: HumanTableTalkInput) -> ExternalInputResult:
        """Build a table-talk result without publishing side effects."""
        speech = _fallback_table_talk_reply(request.target_agent_id)
        reaction: dict[str, object] = {"intent": "table_talk_reply"}
        emotion = "calm"
        if self.agent_banter_source is not None:
            try:
                decision = await self.agent_banter_source.respond_to_human_table_talk(
                    request,
                    self.orchestrator.public_state(),
                )
            except Exception:  # noqa: BLE001
                decision = None
            if decision is not None and decision.speech:
                speech = decision.speech
                reaction = decision.reaction
                emotion = decision.emotion
        return self.orchestrator.submit_human_table_talk(request, speech=speech, reaction=reaction, emotion=emotion)

    async def _maybe_queue_human_action_reaction(self, events: list[GameEvent]) -> list[GameEvent]:
        if self.agent_banter_source is None:
            return []
        action_event = next(
            (
                event
                for event in events
                if event.event_type == EventType.ACTION_COMMITTED and event.payload.get("seat") == 1
            ),
            None,
        )
        if action_event is None:
            return []
        try:
            decision = await self.agent_banter_source.react_to_human_action(
                action_event,
                self.orchestrator.public_state(),
            )
        except Exception:  # noqa: BLE001
            return []
        if decision.agent_id is None or not decision.speech:
            return []
        event = self.orchestrator.record_agent_banter_response(
            decision.agent_id,
            decision.speech,
            reaction=decision.reaction,
            intent="human_action_reaction",
            emotion=decision.emotion,
        )
        return [event]

    async def handle_new_events(self, events: list[GameEvent]) -> None:
        """Handle side effects for newly appended orchestrator events."""
        for event in events:
            if _is_orchestrator_speech_event(event):
                self._queue_orchestrator_synthesis(event.event_id)

    async def synthesize_orchestrator_event(self, event_id: str) -> bytes:
        """Return MPEG audio for one queued orchestrator speech event."""
        cache_key = _audio_cache_key("orchestrator", event_id)
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        task = self._queue_orchestrator_synthesis(event_id)
        if task is None:
            return self._audio_cache[cache_key]
        return await task

    async def synthesize_eliza_event(self, event_id: str) -> bytes:
        """Return MPEG audio for one queued Eliza speech event."""
        cache_key = _audio_cache_key("eliza", event_id)
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        task = self._queue_eliza_synthesis(event_id)
        if task is None:
            return self._audio_cache[cache_key]
        return await task

    def _queue_orchestrator_synthesis(self, event_id: str) -> asyncio.Task[bytes] | None:
        """Start synthesis for a speech event unless audio is already ready."""
        cache_key = _audio_cache_key("orchestrator", event_id)
        if cache_key in self._audio_cache:
            return None
        if cache_key in self._audio_tasks:
            return self._audio_tasks[cache_key]

        task = asyncio.create_task(self._synthesize_orchestrator_event(event_id))
        self._audio_tasks[cache_key] = task
        task.add_done_callback(lambda completed: self._finalize_audio_task(cache_key, completed))
        return task

    def _queue_eliza_synthesis(self, event_id: str) -> asyncio.Task[bytes] | None:
        """Start Eliza speech synthesis unless audio is already ready."""
        cache_key = _audio_cache_key("eliza", event_id)
        if cache_key in self._audio_cache:
            return None
        if cache_key in self._audio_tasks:
            return self._audio_tasks[cache_key]

        task = asyncio.create_task(self._synthesize_eliza_event(event_id))
        self._audio_tasks[cache_key] = task
        task.add_done_callback(lambda completed: self._finalize_audio_task(cache_key, completed))
        return task

    def _finalize_audio_task(self, cache_key: str, task: asyncio.Task[bytes]) -> None:
        """Forget completed synthesis tasks while keeping successful audio cached."""
        if self._audio_tasks.get(cache_key) is task:
            self._audio_tasks.pop(cache_key, None)
        with suppress(asyncio.CancelledError, Exception):
            task.result()

    async def _synthesize_orchestrator_event(self, event_id: str) -> bytes:
        """Synthesize and cache one orchestrator speech event."""
        cache_key = _audio_cache_key("orchestrator", event_id)
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]

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
        self._audio_cache[cache_key] = audio
        return audio

    async def _synthesize_eliza_event(self, event_id: str) -> bytes:
        """Synthesize and cache one Eliza speech event."""
        cache_key = _audio_cache_key("eliza", event_id)
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]

        event = self.orchestrator.event_by_id(event_id)
        if event is None or event.event_type != EventType.PRESENTATION_COMMAND:
            msg = "Unknown Eliza speech event."
            raise ElevenLabsClientError(msg)

        speech = event.payload.get("speech")
        target_client = event.payload.get("target_client")
        if not isinstance(speech, str) or str(target_client) != ClientId.ELIZA:
            msg = "Event does not contain Eliza speech."
            raise ElevenLabsClientError(msg)

        if self._voice_client is None:
            self._voice_client = self.voice_client_factory()
        audio = await self._voice_client.synthesize_eliza(speech)
        self._audio_cache[cache_key] = audio
        return audio


def _public_error_message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def _fallback_table_talk_reply(agent_id: str) -> str:
    display_name = {"reachy": "Reachy", "eliza": "Eliza"}.get(agent_id, "I")
    return f"{display_name} heard you."


def _is_orchestrator_speech_event(event: GameEvent) -> bool:
    return (
        event.event_type == EventType.PRESENTATION_COMMAND
        and event.payload.get("voice") == "orchestrator"
        and isinstance(event.payload.get("speech"), str)
    )


def _audio_cache_key(voice: str, event_id: str) -> str:
    return f"{voice}:{event_id}"
