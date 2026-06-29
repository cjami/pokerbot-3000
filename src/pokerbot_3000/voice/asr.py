"""ElevenLabs speech transcription adapter."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import uuid
import wave
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from pokerbot_3000.ports.voice import AudioChunk, VoiceTranscript
from pokerbot_3000.voice.capture import VoiceRuntimeError
from pokerbot_3000.voice.elevenlabs import (
    DEFAULT_ELEVENLABS_BASE_URL,
    DEFAULT_TIMEOUT_SECONDS,
    ELEVENLABS_API_KEY_ENV,
    ELEVENLABS_BASE_URL_ENV,
    ELEVENLABS_TIMEOUT_ENV,
    ElevenLabsClientError,
    ElevenLabsConfigurationError,
)

if TYPE_CHECKING:
    from pathlib import Path

type SpeechToTextTransport = Callable[[str, bytes, Mapping[str, str], float], Mapping[str, object]]

ELEVENLABS_STT_MODEL_ENV: Final = "ELEVENLABS_STT_MODEL"
ELEVENLABS_STT_LANGUAGE_ENV: Final = "ELEVENLABS_STT_LANGUAGE"
DEFAULT_ELEVENLABS_STT_MODEL: Final = "scribe_v1"
DEFAULT_ELEVENLABS_STT_LANGUAGE: Final = "en"
PCM_SAMPLE_WIDTH_BYTES: Final = 2

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ElevenLabsSpeechTranscriptionConfig:
    """ElevenLabs speech-to-text settings."""

    api_key: str
    model: str = DEFAULT_ELEVENLABS_STT_MODEL
    language: str | None = DEFAULT_ELEVENLABS_STT_LANGUAGE
    base_url: str = DEFAULT_ELEVENLABS_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> ElevenLabsSpeechTranscriptionConfig:
        """Load speech-to-text settings from environment variables."""
        load_dotenv(dotenv_path=env_file)

        api_key = os.getenv(ELEVENLABS_API_KEY_ENV)
        if not api_key:
            msg = f"Set {ELEVENLABS_API_KEY_ENV} in your environment or .env file."
            raise ElevenLabsConfigurationError(msg)

        return cls(
            api_key=api_key,
            model=os.getenv(ELEVENLABS_STT_MODEL_ENV, DEFAULT_ELEVENLABS_STT_MODEL),
            language=_env_optional(ELEVENLABS_STT_LANGUAGE_ENV, DEFAULT_ELEVENLABS_STT_LANGUAGE),
            base_url=_normalized_base_url(os.getenv(ELEVENLABS_BASE_URL_ENV, DEFAULT_ELEVENLABS_BASE_URL)),
            timeout_seconds=_env_float(ELEVENLABS_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS),
        )


class ElevenLabsSpeechTranscriber:
    """Transcribe speech segments with ElevenLabs Scribe."""

    def __init__(
        self,
        config: ElevenLabsSpeechTranscriptionConfig | None = None,
        transport: SpeechToTextTransport | None = None,
    ) -> None:
        """Create an ElevenLabs speech-to-text transcriber."""
        self._config = config
        self._transport = transport or _urllib_speech_to_text_transport

    async def transcribe(self, segment: AudioChunk) -> VoiceTranscript:
        """Transcribe one speech segment."""
        return await asyncio.to_thread(self._transcribe_sync, segment)

    def _transcribe_sync(self, segment: AudioChunk) -> VoiceTranscript:
        config = self._config or ElevenLabsSpeechTranscriptionConfig.from_env()
        audio = _wav_bytes(segment)
        LOGGER.info("Sending voice phrase to ElevenLabs speech-to-text (%d WAV bytes).", len(audio))
        body, content_type = _multipart_body(config, audio)
        headers = {
            "Accept": "application/json",
            "Content-Type": content_type,
            "User-Agent": "pokerbot-3000/0.1",
            "xi-api-key": config.api_key,
        }
        result = self._transport(
            urljoin(config.base_url, "speech-to-text"),
            body,
            headers,
            config.timeout_seconds,
        )
        text = _extract_text(result).strip()
        LOGGER.info("ElevenLabs speech-to-text transcript: %r.", text)
        return VoiceTranscript(text=text, confidence=1.0)


def _wav_bytes(segment: AudioChunk) -> bytes:
    if segment.sample_width != PCM_SAMPLE_WIDTH_BYTES:
        msg = f"Expected 16-bit PCM audio, received sample width {segment.sample_width}."
        raise VoiceRuntimeError(msg)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(segment.sample_width)
        wav_file.setframerate(segment.sample_rate)
        wav_file.writeframes(segment.pcm)
    return buffer.getvalue()


def _multipart_body(config: ElevenLabsSpeechTranscriptionConfig, audio: bytes) -> tuple[bytes, str]:
    fields = {"model_id": config.model}
    if config.language:
        fields["language_code"] = config.language

    boundary = f"pokerbot-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(f"{value}\r\n".encode())
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="file"; filename="speech.wav"\r\n')
    body.extend(b"Content-Type: audio/wav\r\n\r\n")
    body.extend(audio)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _urllib_speech_to_text_transport(
    url: str,
    body: bytes,
    headers: Mapping[str, str],
    timeout: float,
) -> Mapping[str, object]:
    request = Request(url=url, data=body, headers=dict(headers), method="POST")  # noqa: S310
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        msg = f"ElevenLabs speech-to-text failed with HTTP {exc.code}: {_trim(detail)}"
        raise ElevenLabsClientError(msg) from exc
    except URLError as exc:
        msg = f"Could not reach ElevenLabs speech-to-text: {exc.reason}"
        raise ElevenLabsClientError(msg) from exc

    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        msg = f"ElevenLabs speech-to-text returned invalid JSON: {_trim(payload)}"
        raise ElevenLabsClientError(msg) from exc
    if not isinstance(result, dict):
        msg = "ElevenLabs speech-to-text returned an unexpected response shape."
        raise ElevenLabsClientError(msg)
    return result


def _extract_text(result: object) -> str:
    for candidate in _transcription_candidates(result):
        text = str(getattr(candidate, "text", candidate)).strip()
        if text:
            return text
    return ""


def _transcription_candidates(result: object) -> list[object]:
    if isinstance(result, list | tuple):
        candidates: list[object] = []
        for item in result:
            candidates.extend(_transcription_candidates(item))
        return candidates
    if isinstance(result, dict):
        candidates: list[object] = []
        for key in ("text", "transcript"):
            if text := result.get(key):
                candidates.append(text)
        for key in ("chunks", "segments", "results"):
            nested = result.get(key)
            if nested is not None:
                candidates.extend(_transcription_candidates(nested))
        return candidates
    return [result]


def _normalized_base_url(value: str) -> str:
    return value if value.endswith("/") else f"{value}/"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        msg = f"{name} must be a number of seconds."
        raise ElevenLabsConfigurationError(msg) from exc


def _env_optional(name: str, default: str) -> str | None:
    value = os.getenv(name, default).strip()
    return value or None


def _trim(value: str, max_length: int = 500) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length]}..."
