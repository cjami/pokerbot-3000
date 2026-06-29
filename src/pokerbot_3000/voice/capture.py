"""Microphone capture and voice activity detection adapters."""

from __future__ import annotations

import array
import asyncio
import importlib
import logging
import math
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pokerbot_3000.ports.voice import AudioChunk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

POKERBOT_VOICE_DEVICE_ENV: Final = "POKERBOT_VOICE_DEVICE"
POKERBOT_VAD_RMS_THRESHOLD_ENV: Final = "POKERBOT_VAD_RMS_THRESHOLD"
POKERBOT_VAD_MIN_PHRASE_MS_ENV: Final = "POKERBOT_VAD_MIN_PHRASE_MS"
POKERBOT_VAD_MAX_PHRASE_MS_ENV: Final = "POKERBOT_VAD_MAX_PHRASE_MS"
POKERBOT_VAD_SILENCE_MS_ENV: Final = "POKERBOT_VAD_SILENCE_MS"

DEFAULT_SAMPLE_RATE: Final = 16_000
DEFAULT_BLOCK_SIZE: Final = 512
DEFAULT_VAD_RMS_THRESHOLD: Final = 0.012
DEFAULT_MIN_PHRASE_MS: Final = 220
DEFAULT_MAX_PHRASE_MS: Final = 8_000
DEFAULT_SILENCE_MS: Final = 650

LOGGER = logging.getLogger(__name__)


class VoiceRuntimeError(RuntimeError):
    """Raised when the local voice runtime cannot initialize or process audio."""


@dataclass(frozen=True, slots=True)
class MicrophoneConfig:
    """Local microphone capture settings."""

    device: str | int | None = None
    sample_rate: int = DEFAULT_SAMPLE_RATE
    block_size: int = DEFAULT_BLOCK_SIZE

    @classmethod
    def from_env(cls) -> MicrophoneConfig:
        """Load microphone settings from environment variables."""
        raw_device = os.getenv(POKERBOT_VOICE_DEVICE_ENV)
        device: str | int | None = None
        if raw_device:
            device = int(raw_device) if raw_device.isdigit() else raw_device
        return cls(device=device)


@dataclass(frozen=True, slots=True)
class VadConfig:
    """Voice phrase segmentation settings."""

    rms_threshold: float = DEFAULT_VAD_RMS_THRESHOLD
    min_phrase_ms: int = DEFAULT_MIN_PHRASE_MS
    max_phrase_ms: int = DEFAULT_MAX_PHRASE_MS
    silence_ms: int = DEFAULT_SILENCE_MS

    @classmethod
    def from_env(cls) -> VadConfig:
        """Load VAD settings from environment variables."""
        return cls(
            rms_threshold=float(os.getenv(POKERBOT_VAD_RMS_THRESHOLD_ENV, DEFAULT_VAD_RMS_THRESHOLD)),
            min_phrase_ms=int(os.getenv(POKERBOT_VAD_MIN_PHRASE_MS_ENV, DEFAULT_MIN_PHRASE_MS)),
            max_phrase_ms=int(os.getenv(POKERBOT_VAD_MAX_PHRASE_MS_ENV, DEFAULT_MAX_PHRASE_MS)),
            silence_ms=int(os.getenv(POKERBOT_VAD_SILENCE_MS_ENV, DEFAULT_SILENCE_MS)),
        )


class SoundDeviceAudioInput:
    """Capture mono 16-bit PCM chunks from a local microphone."""

    def __init__(self, config: MicrophoneConfig | None = None) -> None:
        """Create a microphone input adapter."""
        self._config = config or MicrophoneConfig.from_env()

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        """Yield microphone chunks from sounddevice until the consumer stops."""
        try:
            sd = importlib.import_module("sounddevice")
        except ImportError as exc:  # pragma: no cover - environment dependent
            msg = "Install sounddevice to use microphone voice capture."
            raise VoiceRuntimeError(msg) from exc

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)

        def callback(indata: bytes, _frames: int, _time: object, _status: object) -> None:
            loop.call_soon_threadsafe(_put_nowait_drop_oldest, queue, bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=self._config.sample_rate,
                blocksize=self._config.block_size,
                channels=1,
                dtype="int16",
                device=self._config.device,
                callback=callback,
            ):
                while True:
                    yield AudioChunk(pcm=await queue.get(), sample_rate=self._config.sample_rate)
        except Exception as exc:  # pragma: no cover - environment dependent
            msg = f"Microphone capture failed: {exc}"
            raise VoiceRuntimeError(msg) from exc


class BrowserAudioInput:
    """Receive mono 16-bit PCM chunks submitted by the browser."""

    def __init__(self, *, queue_size: int = 128, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        """Create a browser-fed audio queue."""
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_size)
        self._sample_rate = sample_rate
        self._connection_count = 0
        self._submitted_chunk_count = 0
        self._submitted_byte_count = 0

    @property
    def connected(self) -> bool:
        """Return whether at least one browser is streaming microphone audio."""
        return self._connection_count > 0

    @property
    def pending_chunk_count(self) -> int:
        """Return the number of queued browser audio chunks."""
        return self._queue.qsize()

    @property
    def submitted_chunk_count(self) -> int:
        """Return the number of browser chunks received."""
        return self._submitted_chunk_count

    @property
    def submitted_byte_count(self) -> int:
        """Return the number of browser audio bytes received."""
        return self._submitted_byte_count

    def discard_pending(self) -> int:
        """Drop queued browser audio chunks and return how many were discarded."""
        discarded = 0
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return discarded
            discarded += 1

    def connect(self) -> None:
        """Record a browser voice stream connection."""
        self._connection_count += 1

    def disconnect(self) -> None:
        """Record a browser voice stream disconnect."""
        self._connection_count = max(0, self._connection_count - 1)

    async def submit_pcm(self, pcm: bytes) -> None:
        """Submit one 16 kHz mono PCM chunk from the browser."""
        if not pcm:
            return
        self._submitted_chunk_count += 1
        self._submitted_byte_count += len(pcm)
        if self._submitted_chunk_count == 1:
            LOGGER.info("Received first browser voice chunk (%d bytes).", len(pcm))
        _put_nowait_drop_oldest(self._queue, pcm)

    async def chunks(self) -> AsyncIterator[AudioChunk]:
        """Yield browser-submitted audio chunks."""
        while True:
            yield AudioChunk(pcm=await self._queue.get(), sample_rate=self._sample_rate)


class EnergyVoiceActivityDetector:
    """Segment browser microphone chunks with RMS energy and trailing silence."""

    def __init__(self, config: VadConfig | None = None) -> None:
        """Create a lightweight phrase segmenter."""
        self._config = config or VadConfig.from_env()

    async def speech_segments(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[AudioChunk]:
        """Yield speech phrases when energy rises and then returns to silence."""
        phrase: list[bytes] = []
        in_speech = False
        phrase_byte_count = 0
        silence_byte_count = 0
        min_phrase_bytes = _millis_to_bytes(self._config.min_phrase_ms)
        max_phrase_bytes = _millis_to_bytes(self._config.max_phrase_ms)
        silence_bytes = _millis_to_bytes(self._config.silence_ms)

        async for chunk in chunks:
            if chunk.sample_rate != DEFAULT_SAMPLE_RATE:
                msg = f"Expected {DEFAULT_SAMPLE_RATE} Hz audio, received {chunk.sample_rate} Hz."
                raise VoiceRuntimeError(msg)

            has_voice = _has_voice_energy(chunk.pcm, self._config.rms_threshold)
            if has_voice:
                if not in_speech:
                    LOGGER.info("Voice activity started.")
                    phrase = []
                    phrase_byte_count = 0
                    silence_byte_count = 0
                in_speech = True
                silence_byte_count = 0
            elif in_speech:
                silence_byte_count += len(chunk.pcm)

            if not in_speech:
                continue

            phrase.append(chunk.pcm)
            phrase_byte_count += len(chunk.pcm)
            should_end = silence_byte_count >= silence_bytes or phrase_byte_count >= max_phrase_bytes
            if not should_end:
                continue

            in_speech = False
            if phrase_byte_count >= min_phrase_bytes:
                LOGGER.info("Voice phrase completed (%d bytes).", phrase_byte_count)
                yield AudioChunk(pcm=b"".join(phrase), sample_rate=chunk.sample_rate)
            phrase = []
            phrase_byte_count = 0
            silence_byte_count = 0


def _put_nowait_drop_oldest(queue: asyncio.Queue[bytes], data: bytes) -> None:
    if queue.full():
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    with suppress(asyncio.QueueFull):
        queue.put_nowait(data)


def _millis_to_bytes(milliseconds: int) -> int:
    return int(DEFAULT_SAMPLE_RATE * 2 * (milliseconds / 1000))


def _has_voice_energy(pcm: bytes, rms_threshold: float) -> bool:
    samples = array.array("h")
    samples.frombytes(pcm)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return False

    peak = max(abs(sample) for sample in samples) / 32768.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    rms = math.sqrt(mean_square) / 32768.0
    return rms >= rms_threshold or peak >= max(0.08, rms_threshold * 4)
