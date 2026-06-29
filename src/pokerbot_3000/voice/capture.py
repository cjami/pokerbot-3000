"""Microphone capture and voice activity detection adapters."""

from __future__ import annotations

import asyncio
import importlib
import os
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pokerbot_3000.ports.voice import AudioChunk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

POKERBOT_VOICE_DEVICE_ENV: Final = "POKERBOT_VOICE_DEVICE"
POKERBOT_VAD_THRESHOLD_ENV: Final = "POKERBOT_VAD_THRESHOLD"
POKERBOT_VAD_MIN_PHRASE_MS_ENV: Final = "POKERBOT_VAD_MIN_PHRASE_MS"
POKERBOT_VAD_MAX_PHRASE_MS_ENV: Final = "POKERBOT_VAD_MAX_PHRASE_MS"

DEFAULT_SAMPLE_RATE: Final = 16_000
DEFAULT_BLOCK_SIZE: Final = 512
DEFAULT_VAD_THRESHOLD: Final = 0.5
DEFAULT_MIN_PHRASE_MS: Final = 220
DEFAULT_MAX_PHRASE_MS: Final = 8_000


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
    """Silero phrase segmentation settings."""

    threshold: float = DEFAULT_VAD_THRESHOLD
    min_phrase_ms: int = DEFAULT_MIN_PHRASE_MS
    max_phrase_ms: int = DEFAULT_MAX_PHRASE_MS

    @classmethod
    def from_env(cls) -> VadConfig:
        """Load VAD settings from environment variables."""
        return cls(
            threshold=float(os.getenv(POKERBOT_VAD_THRESHOLD_ENV, DEFAULT_VAD_THRESHOLD)),
            min_phrase_ms=int(os.getenv(POKERBOT_VAD_MIN_PHRASE_MS_ENV, DEFAULT_MIN_PHRASE_MS)),
            max_phrase_ms=int(os.getenv(POKERBOT_VAD_MAX_PHRASE_MS_ENV, DEFAULT_MAX_PHRASE_MS)),
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


class SileroVoiceActivityDetector:
    """Segment microphone chunks into speech phrases with Silero VAD."""

    def __init__(self, config: VadConfig | None = None) -> None:
        """Create a Silero-backed VAD adapter."""
        self._config = config or VadConfig.from_env()

    async def speech_segments(self, chunks: AsyncIterator[AudioChunk]) -> AsyncIterator[AudioChunk]:
        """Yield full speech phrases from raw microphone chunks."""
        try:
            np = importlib.import_module("numpy")
            torch = importlib.import_module("torch")
            silero_vad = importlib.import_module("silero_vad")
        except ImportError as exc:  # pragma: no cover - environment dependent
            msg = "Install numpy, torch, and silero-vad to use voice activity detection."
            raise VoiceRuntimeError(msg) from exc

        model = silero_vad.load_silero_vad()
        vad_iterator = silero_vad.VADIterator(
            model,
            threshold=self._config.threshold,
            sampling_rate=DEFAULT_SAMPLE_RATE,
        )
        phrase: list[bytes] = []
        in_speech = False
        min_phrase_bytes = _millis_to_bytes(self._config.min_phrase_ms)
        max_phrase_bytes = _millis_to_bytes(self._config.max_phrase_ms)

        async for chunk in chunks:
            if chunk.sample_rate != DEFAULT_SAMPLE_RATE:
                msg = f"Expected {DEFAULT_SAMPLE_RATE} Hz audio, received {chunk.sample_rate} Hz."
                raise VoiceRuntimeError(msg)

            samples = np.frombuffer(chunk.pcm, dtype=np.int16).astype(np.float32) / 32768.0
            event = vad_iterator(torch.from_numpy(samples), return_seconds=False)
            if event and "start" in event:
                in_speech = True
                phrase = []

            if in_speech:
                phrase.append(chunk.pcm)

            phrase_bytes = b"".join(phrase)
            should_end = bool(event and "end" in event) or len(phrase_bytes) >= max_phrase_bytes
            if in_speech and should_end:
                in_speech = False
                if len(phrase_bytes) >= min_phrase_bytes:
                    yield AudioChunk(pcm=phrase_bytes, sample_rate=chunk.sample_rate)
                phrase = []


def _put_nowait_drop_oldest(queue: asyncio.Queue[bytes], data: bytes) -> None:
    if queue.full():
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
    with suppress(asyncio.QueueFull):
        queue.put_nowait(data)


def _millis_to_bytes(milliseconds: int) -> int:
    return int(DEFAULT_SAMPLE_RATE * 2 * (milliseconds / 1000))
