"""ElevenLabs-backed speech synthesis."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from dotenv import load_dotenv

if TYPE_CHECKING:
    from pathlib import Path

type AudioTransport = Callable[[str, dict[str, object], Mapping[str, str], float], bytes]

ELEVENLABS_API_KEY_ENV: Final = "ELEVENLABS_API_KEY"
ELEVENLABS_ORCHESTRATOR_VOICE_ID_ENV: Final = "ELEVENLABS_ORCHESTRATOR_VOICE_ID"
ELEVENLABS_ELIZA_VOICE_ID_ENV: Final = "ELEVENLABS_ELIZA_VOICE_ID"
ELEVENLABS_REACHY_VOICE_ID_ENV: Final = "ELEVENLABS_REACHY_VOICE_ID"
ELEVENLABS_MODEL_ENV: Final = "ELEVENLABS_MODEL"
ELEVENLABS_BASE_URL_ENV: Final = "ELEVENLABS_BASE_URL"
ELEVENLABS_TIMEOUT_ENV: Final = "ELEVENLABS_TIMEOUT_SECONDS"
ELEVENLABS_ORCHESTRATOR_SPEED_ENV: Final = "ELEVENLABS_ORCHESTRATOR_SPEED"
ELEVENLABS_ELIZA_SPEED_ENV: Final = "ELEVENLABS_ELIZA_SPEED"
ELEVENLABS_REACHY_SPEED_ENV: Final = "ELEVENLABS_REACHY_SPEED"
DEFAULT_ELEVENLABS_MODEL: Final = "eleven_flash_v2_5"
DEFAULT_ELEVENLABS_BASE_URL: Final = "https://api.elevenlabs.io/v1/"
DEFAULT_OUTPUT_FORMAT: Final = "mp3_44100_128"
DEFAULT_TIMEOUT_SECONDS: Final = 20.0
DEFAULT_ORCHESTRATOR_SPEED: Final = 0.82
DEFAULT_ORCHESTRATOR_STABILITY: Final = 0.55
DEFAULT_ORCHESTRATOR_SIMILARITY_BOOST: Final = 0.8


class ElevenLabsConfigurationError(RuntimeError):
    """Raised when ElevenLabs configuration is missing or invalid."""


class ElevenLabsClientError(RuntimeError):
    """Raised when ElevenLabs speech synthesis fails."""


@dataclass(frozen=True, slots=True)
class ElevenLabsConfig:
    """Runtime configuration for ElevenLabs text-to-speech."""

    api_key: str
    orchestrator_voice_id: str
    eliza_voice_id: str | None = None
    reachy_voice_id: str | None = None
    model: str = DEFAULT_ELEVENLABS_MODEL
    base_url: str = DEFAULT_ELEVENLABS_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    output_format: str = DEFAULT_OUTPUT_FORMAT
    orchestrator_speed: float = DEFAULT_ORCHESTRATOR_SPEED
    eliza_speed: float = DEFAULT_ORCHESTRATOR_SPEED
    reachy_speed: float = DEFAULT_ORCHESTRATOR_SPEED

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> ElevenLabsConfig:
        """Load ElevenLabs settings from environment variables and an optional dotenv file."""
        load_dotenv(dotenv_path=env_file)

        api_key = os.getenv(ELEVENLABS_API_KEY_ENV)
        if not api_key:
            msg = f"Set {ELEVENLABS_API_KEY_ENV} in your environment or .env file."
            raise ElevenLabsConfigurationError(msg)

        voice_id = os.getenv(ELEVENLABS_ORCHESTRATOR_VOICE_ID_ENV)
        if not voice_id:
            msg = f"Set {ELEVENLABS_ORCHESTRATOR_VOICE_ID_ENV} in your environment or .env file."
            raise ElevenLabsConfigurationError(msg)

        return cls(
            api_key=api_key,
            orchestrator_voice_id=voice_id,
            eliza_voice_id=os.getenv(ELEVENLABS_ELIZA_VOICE_ID_ENV),
            reachy_voice_id=os.getenv(ELEVENLABS_REACHY_VOICE_ID_ENV),
            model=os.getenv(ELEVENLABS_MODEL_ENV, DEFAULT_ELEVENLABS_MODEL),
            base_url=_normalized_base_url(os.getenv(ELEVENLABS_BASE_URL_ENV, DEFAULT_ELEVENLABS_BASE_URL)),
            timeout_seconds=_env_float(ELEVENLABS_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS),
            orchestrator_speed=_env_float(ELEVENLABS_ORCHESTRATOR_SPEED_ENV, DEFAULT_ORCHESTRATOR_SPEED),
            eliza_speed=_env_float(ELEVENLABS_ELIZA_SPEED_ENV, DEFAULT_ORCHESTRATOR_SPEED),
            reachy_speed=_env_float(ELEVENLABS_REACHY_SPEED_ENV, DEFAULT_ORCHESTRATOR_SPEED),
        )


class ElevenLabsClient:
    """Small ElevenLabs text-to-speech client."""

    def __init__(self, config: ElevenLabsConfig, transport: AudioTransport | None = None) -> None:
        """Create a client with injectable transport for tests."""
        self._config = config
        self._transport = transport or _urllib_audio_transport

    async def synthesize_orchestrator(self, text: str) -> bytes:
        """Synthesize orchestrator speech as MPEG audio."""
        return await self._synthesize(self._config.orchestrator_voice_id, text, speed=self._config.orchestrator_speed)

    async def synthesize_eliza(self, text: str) -> bytes:
        """Synthesize Eliza speech as MPEG audio."""
        if not self._config.eliza_voice_id:
            msg = f"Set {ELEVENLABS_ELIZA_VOICE_ID_ENV} in your environment or .env file."
            raise ElevenLabsConfigurationError(msg)
        return await self._synthesize(self._config.eliza_voice_id, text, speed=self._config.eliza_speed)

    async def synthesize_reachy(self, text: str) -> bytes:
        """Synthesize Reachy speech as MPEG audio."""
        if not self._config.reachy_voice_id:
            msg = f"Set {ELEVENLABS_REACHY_VOICE_ID_ENV} in your environment or .env file."
            raise ElevenLabsConfigurationError(msg)
        return await self._synthesize(self._config.reachy_voice_id, text, speed=self._config.reachy_speed)

    async def _synthesize(self, voice_id: str, text: str, *, speed: float) -> bytes:
        if not text.strip():
            msg = "Cannot synthesize empty speech."
            raise ElevenLabsClientError(msg)

        query = urlencode({"output_format": self._config.output_format})
        url = urljoin(self._config.base_url, f"text-to-speech/{voice_id}?{query}")
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "User-Agent": "pokerbot-3000/0.1",
            "xi-api-key": self._config.api_key,
        }
        payload: dict[str, object] = {
            "text": text,
            "model_id": self._config.model,
            "voice_settings": {
                "stability": DEFAULT_ORCHESTRATOR_STABILITY,
                "similarity_boost": DEFAULT_ORCHESTRATOR_SIMILARITY_BOOST,
                "speed": speed,
            },
        }
        return await asyncio.to_thread(self._transport, url, payload, headers, self._config.timeout_seconds)


def _urllib_audio_transport(
    url: str,
    payload: dict[str, object],
    headers: Mapping[str, str],
    timeout: float,
) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=data, headers=dict(headers), method="POST")  # noqa: S310
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        msg = f"ElevenLabs API request failed with HTTP {exc.code}: {_trim(detail)}"
        raise ElevenLabsClientError(msg) from exc
    except URLError as exc:
        msg = f"Could not reach ElevenLabs API: {exc.reason}"
        raise ElevenLabsClientError(msg) from exc


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


def _trim(value: str, max_length: int = 500) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length]}..."
