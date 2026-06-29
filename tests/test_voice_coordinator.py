import asyncio
from collections.abc import AsyncIterator
from typing import Any

from pokerbot_3000.domain.cards import Card
from pokerbot_3000.domain.models import PendingInputType, PrivateCardObservation, PublicTableObservation, Street
from pokerbot_3000.orchestrator import InMemoryOrchestrator
from pokerbot_3000.ports.voice import AudioChunk, VoiceTranscript
from pokerbot_3000.voice import DeterministicVoiceCommandParser, VoiceActionAdapters, VoiceActionCoordinator


class _OneShotAudioInput:
    def __init__(self) -> None:
        self.chunk = AudioChunk(pcm=b"speech")

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        yield self.chunk


class _PassThroughVad:
    async def speech_segments(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[AudioChunk]:
        async for chunk in chunks:
            yield chunk


class _StaticTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def transcribe(self, segment: AudioChunk) -> VoiceTranscript:
        self.calls += 1
        assert segment.pcm == b"speech"
        return VoiceTranscript(text=self.text, confidence=0.93)


def test_voice_coordinator_submits_parsed_action_when_waiting_for_human():
    async def scenario() -> None:
        orchestrator = InMemoryOrchestrator()
        orchestrator.start_game()
        _open_human_action_after_flop(orchestrator)
        events_seen: list[str] = []
        publishes = 0

        async def after_events(events: list[Any]) -> None:
            events_seen.extend(event.event_type for event in events)

        async def publish() -> None:
            nonlocal publishes
            publishes += 1

        coordinator = VoiceActionCoordinator(
            orchestrator=orchestrator,
            adapters=VoiceActionAdapters(
                audio_input=_OneShotAudioInput(),
                vad=_PassThroughVad(),
                transcriber=_StaticTranscriber("bet one hundred"),
                parser=DeterministicVoiceCommandParser(),
            ),
            after_events=after_events,
            publish_snapshot=publish,
        )

        coordinator.start()
        await _wait_for(lambda: coordinator.status.latest_action is not None)
        await coordinator.stop()

        assert orchestrator.public_state().pot == 300
        assert "action_proposed" in events_seen
        assert "action_committed" in events_seen
        assert coordinator.status.latest_action == {"type": "bet", "amount": 100, "unit": "chips"}
        assert publishes > 0

    asyncio.run(scenario())


def test_voice_coordinator_ignores_segments_when_not_waiting_for_human():
    async def scenario() -> None:
        orchestrator = InMemoryOrchestrator()
        orchestrator.start_game()
        transcriber = _StaticTranscriber("check")
        coordinator = VoiceActionCoordinator(
            orchestrator=orchestrator,
            adapters=VoiceActionAdapters(
                audio_input=_OneShotAudioInput(),
                vad=_PassThroughVad(),
                transcriber=transcriber,
                parser=DeterministicVoiceCommandParser(),
            ),
        )

        coordinator.start()
        await asyncio.sleep(0)
        await coordinator.stop()

        assert transcriber.calls == 0
        waiting_for = orchestrator.public_state().waiting_for
        assert waiting_for is not None
        assert waiting_for.type == PendingInputType.PUBLIC_BOARD_CARDS

    asyncio.run(scenario())


async def _wait_for(predicate) -> None:
    for _ in range(20):
        if predicate():
            return
        await asyncio.sleep(0.01)
    msg = "Timed out waiting for coordinator state."
    raise AssertionError(msg)


def _open_human_action_after_flop(orchestrator: InMemoryOrchestrator) -> None:
    flop = [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")]
    for _ in range(2):
        orchestrator.record_public_observation(
            PublicTableObservation(board_cards=flop, street_hint=Street.FLOP, confidence=0.9)
        )
    orchestrator.record_client_private_cards(
        "eliza",
        PrivateCardObservation(
            agent_id="eliza",
            seat=3,
            hole_cards=[_card("9", "clubs"), _card("9", "diamonds")],
            source="eliza_browser_webcam",
            confidence=0.89,
        ),
    )
    orchestrator.record_client_private_cards(
        "reachy",
        PrivateCardObservation(
            agent_id="reachy",
            seat=2,
            hole_cards=[_card("king", "clubs"), _card("king", "diamonds")],
            source="reachy_camera",
            confidence=0.89,
        ),
    )


def _card(rank: str, suit: str) -> Card:
    return Card.model_validate({"rank": rank, "suit": suit})
