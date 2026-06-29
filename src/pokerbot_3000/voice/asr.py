"""Parakeet speech transcription adapter."""

from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

from pokerbot_3000.ports.voice import AudioChunk, VoiceTranscript
from pokerbot_3000.voice.capture import VoiceRuntimeError

POKERBOT_VOICE_MODEL_ENV: Final = "POKERBOT_VOICE_MODEL"
DEFAULT_PARAKEET_MODEL: Final = "nvidia/parakeet-unified-en-0.6b"


class _ParakeetModel(Protocol):
    def transcribe(self, paths: list[str]) -> object:
        """Transcribe audio files."""


@dataclass(frozen=True, slots=True)
class ParakeetConfig:
    """NVIDIA Parakeet ASR settings."""

    model_name: str = DEFAULT_PARAKEET_MODEL

    @classmethod
    def from_env(cls) -> ParakeetConfig:
        """Load ASR settings from environment variables."""
        return cls(model_name=os.getenv(POKERBOT_VOICE_MODEL_ENV, DEFAULT_PARAKEET_MODEL))


class ParakeetSpeechTranscriber:
    """Transcribe speech segments with NVIDIA Parakeet through NeMo."""

    def __init__(self, config: ParakeetConfig | None = None) -> None:
        """Create a lazy Parakeet transcriber."""
        self._config = config or ParakeetConfig.from_env()
        self._model: _ParakeetModel | None = None

    async def transcribe(self, segment: AudioChunk) -> VoiceTranscript:
        """Transcribe one speech segment."""
        return await asyncio.to_thread(self._transcribe_sync, segment)

    def _transcribe_sync(self, segment: AudioChunk) -> VoiceTranscript:
        model = self._load_model()
        with tempfile.TemporaryDirectory(prefix="pokerbot_voice_") as directory:
            wav_path = Path(directory) / "speech.wav"
            _write_wav(wav_path, segment)
            result = model.transcribe([str(wav_path)])
        text = _extract_text(result)
        return VoiceTranscript(text=text, confidence=1.0)

    def _load_model(self) -> _ParakeetModel:
        if self._model is not None:
            return self._model
        try:
            nemo_asr = importlib.import_module("nemo.collections.asr")
        except ImportError as exc:  # pragma: no cover - environment dependent
            msg = "Install nemo_toolkit[asr] to use NVIDIA Parakeet transcription."
            raise VoiceRuntimeError(msg) from exc
        try:
            self._model = cast(
                "_ParakeetModel",
                nemo_asr.models.ASRModel.from_pretrained(model_name=self._config.model_name),
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            msg = f"Parakeet model initialization failed: {exc}"
            raise VoiceRuntimeError(msg) from exc
        return self._model


def _write_wav(path: Path, segment: AudioChunk) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(segment.sample_width)
        wav_file.setframerate(segment.sample_rate)
        wav_file.writeframes(segment.pcm)


def _extract_text(result: object) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        return str(getattr(first, "text", first)).strip()
    return str(result).strip()
