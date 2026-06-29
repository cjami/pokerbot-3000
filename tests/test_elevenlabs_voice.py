import asyncio
from collections.abc import Mapping
from typing import Any

import pytest

from pokerbot_3000.voice import (
    ELEVENLABS_API_KEY_ENV,
    ELEVENLABS_ELIZA_VOICE_ID_ENV,
    ELEVENLABS_MODEL_ENV,
    ELEVENLABS_ORCHESTRATOR_VOICE_ID_ENV,
    ElevenLabsClient,
    ElevenLabsClientError,
    ElevenLabsConfig,
    ElevenLabsConfigurationError,
)
from pokerbot_3000.voice.elevenlabs import DEFAULT_ELEVENLABS_MODEL


def test_elevenlabs_config_loads_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv(ELEVENLABS_API_KEY_ENV, raising=False)
    monkeypatch.delenv(ELEVENLABS_ELIZA_VOICE_ID_ENV, raising=False)
    monkeypatch.delenv(ELEVENLABS_ORCHESTRATOR_VOICE_ID_ENV, raising=False)
    monkeypatch.delenv(ELEVENLABS_MODEL_ENV, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ELEVENLABS_API_KEY=test-key\n"
        "ELEVENLABS_ORCHESTRATOR_VOICE_ID=test-voice\n"
        "ELEVENLABS_ELIZA_VOICE_ID=eliza-voice\n"
        "ELEVENLABS_MODEL=test-model\n",
        encoding="utf-8",
    )

    config = ElevenLabsConfig.from_env(env_file)

    assert config.api_key == "test-key"
    assert config.orchestrator_voice_id == "test-voice"
    assert config.eliza_voice_id == "eliza-voice"
    assert config.model == "test-model"
    assert config.orchestrator_speed == 0.82


def test_elevenlabs_config_requires_orchestrator_voice_id(tmp_path, monkeypatch):
    monkeypatch.delenv(ELEVENLABS_API_KEY_ENV, raising=False)
    monkeypatch.delenv(ELEVENLABS_ORCHESTRATOR_VOICE_ID_ENV, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("ELEVENLABS_API_KEY=test-key\n", encoding="utf-8")

    with pytest.raises(ElevenLabsConfigurationError, match=ELEVENLABS_ORCHESTRATOR_VOICE_ID_ENV):
        ElevenLabsConfig.from_env(env_file)


def test_elevenlabs_client_uses_flash_model_and_orchestrator_voice():
    calls: list[dict[str, Any]] = []

    def fake_transport(url: str, payload: dict[str, object], headers: Mapping[str, str], _timeout: float) -> bytes:
        calls.append({"url": url, "payload": payload, "headers": dict(headers)})
        return b"mp3-bytes"

    client = ElevenLabsClient(
        ElevenLabsConfig(api_key="test-key", orchestrator_voice_id="voice-123"),
        transport=fake_transport,
    )

    audio = asyncio.run(client.synthesize_orchestrator("Please lay out the flop."))

    assert audio == b"mp3-bytes"
    call = calls[0]
    assert call["url"] == "https://api.elevenlabs.io/v1/text-to-speech/voice-123?output_format=mp3_44100_128"
    assert call["payload"] == {
        "text": "Please lay out the flop.",
        "model_id": DEFAULT_ELEVENLABS_MODEL,
        "voice_settings": {
            "stability": 0.55,
            "similarity_boost": 0.8,
            "speed": 0.82,
        },
    }
    headers = call["headers"]
    assert headers["xi-api-key"] == "test-key"
    assert headers["Accept"] == "audio/mpeg"


def test_elevenlabs_client_rejects_empty_text():
    client = ElevenLabsClient(ElevenLabsConfig(api_key="test-key", orchestrator_voice_id="voice-123"))

    with pytest.raises(ElevenLabsClientError, match="empty"):
        asyncio.run(client.synthesize_orchestrator(" "))


def test_elevenlabs_client_uses_eliza_voice_when_configured():
    calls: list[dict[str, Any]] = []

    def fake_transport(url: str, payload: dict[str, object], headers: Mapping[str, str], _timeout: float) -> bytes:
        calls.append({"url": url, "payload": payload, "headers": dict(headers)})
        return b"eliza-mp3"

    client = ElevenLabsClient(
        ElevenLabsConfig(api_key="test-key", orchestrator_voice_id="voice-123", eliza_voice_id="eliza-456"),
        transport=fake_transport,
    )

    audio = asyncio.run(client.synthesize_eliza("Eliza checks."))

    assert audio == b"eliza-mp3"
    assert calls[0]["url"] == "https://api.elevenlabs.io/v1/text-to-speech/eliza-456?output_format=mp3_44100_128"


def test_elevenlabs_client_requires_eliza_voice_for_eliza_speech():
    client = ElevenLabsClient(ElevenLabsConfig(api_key="test-key", orchestrator_voice_id="voice-123"))

    with pytest.raises(ElevenLabsConfigurationError, match=ELEVENLABS_ELIZA_VOICE_ID_ENV):
        asyncio.run(client.synthesize_eliza("Eliza checks."))
