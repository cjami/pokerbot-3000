import asyncio
from collections.abc import AsyncIterator

from pokerbot_3000.ports.voice import AudioChunk
from pokerbot_3000.voice import BrowserAudioInput, EnergyVoiceActivityDetector, VadConfig


def test_energy_vad_emits_phrase_after_trailing_silence():
    async def scenario() -> list[AudioChunk]:
        detector = EnergyVoiceActivityDetector(
            VadConfig(
                rms_threshold=0.01,
                min_phrase_ms=50,
                max_phrase_ms=2000,
                silence_ms=96,
            )
        )
        return [segment async for segment in detector.speech_segments(_chunks())]

    segments = asyncio.run(scenario())

    assert len(segments) == 1
    assert len(segments[0].pcm) >= 512 * 2 * 4


def test_browser_audio_input_can_discard_pending_chunks():
    async def scenario() -> BrowserAudioInput:
        audio_input = BrowserAudioInput()
        await audio_input.submit_pcm(b"\x00\x01" * 512)
        await audio_input.submit_pcm(b"\x02\x03" * 512)
        return audio_input

    audio_input = asyncio.run(scenario())

    assert audio_input.pending_chunk_count == 2
    assert audio_input.discard_pending() == 2
    assert audio_input.pending_chunk_count == 0
    assert audio_input.discard_pending() == 0


async def _chunks() -> AsyncIterator[AudioChunk]:
    silence = _pcm_chunk(0)
    speech = _pcm_chunk(8000)
    for pcm in [silence, silence, speech, speech, speech, speech, silence, silence, silence]:
        yield AudioChunk(pcm=pcm)


def _pcm_chunk(value: int) -> bytes:
    return value.to_bytes(2, byteorder="little", signed=True) * 512
