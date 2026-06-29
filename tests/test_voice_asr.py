import asyncio
from collections.abc import Mapping

import pytest

from pokerbot_3000.ports.voice import AudioChunk
from pokerbot_3000.voice import (
    DEFAULT_ELEVENLABS_STT_LANGUAGE,
    DEFAULT_ELEVENLABS_STT_MODEL,
    ELEVENLABS_API_KEY_ENV,
    ELEVENLABS_STT_KEYTERMS_ENV,
    ELEVENLABS_STT_LANGUAGE_ENV,
    ELEVENLABS_STT_MODEL_ENV,
    ElevenLabsSpeechTranscriber,
    ElevenLabsSpeechTranscriptionConfig,
    VoiceRuntimeError,
)
from pokerbot_3000.voice.asr import _pcm_bytes
from pokerbot_3000.voice.elevenlabs import ELEVENLABS_BASE_URL_ENV


class _RecordingSpeechToTextTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bytes, Mapping[str, str], float]] = []
        self.result: Mapping[str, object] = {"text": "check"}

    def __call__(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout: float,
    ) -> Mapping[str, object]:
        self.calls.append((url, body, headers, timeout))
        return self.result


def test_elevenlabs_stt_config_uses_default_model_and_language(monkeypatch):
    monkeypatch.setenv(ELEVENLABS_API_KEY_ENV, "test-key")
    monkeypatch.delenv(ELEVENLABS_STT_MODEL_ENV, raising=False)
    monkeypatch.delenv(ELEVENLABS_STT_LANGUAGE_ENV, raising=False)
    monkeypatch.delenv(ELEVENLABS_STT_KEYTERMS_ENV, raising=False)

    config = ElevenLabsSpeechTranscriptionConfig.from_env()

    assert config.model == DEFAULT_ELEVENLABS_STT_MODEL
    assert config.language == DEFAULT_ELEVENLABS_STT_LANGUAGE
    assert "call" in config.keyterms


def test_elevenlabs_stt_config_reads_model_and_language_from_env(monkeypatch):
    monkeypatch.setenv(ELEVENLABS_API_KEY_ENV, "test-key")
    monkeypatch.setenv(ELEVENLABS_STT_MODEL_ENV, "scribe-custom")
    monkeypatch.setenv(ELEVENLABS_STT_LANGUAGE_ENV, "en")
    monkeypatch.setenv(ELEVENLABS_STT_KEYTERMS_ENV, "fold, call, raise")
    monkeypatch.setenv(ELEVENLABS_BASE_URL_ENV, "https://example.test/v1")

    config = ElevenLabsSpeechTranscriptionConfig.from_env()

    assert config.model == "scribe-custom"
    assert config.language == "en"
    assert config.keyterms == ("fold", "call", "raise")
    assert config.base_url == "https://example.test/v1/"


def test_elevenlabs_transcriber_posts_pcm_to_speech_to_text():
    transport = _RecordingSpeechToTextTransport()
    transcriber = ElevenLabsSpeechTranscriber(
        ElevenLabsSpeechTranscriptionConfig(api_key="test-key", model="scribe-custom", language="en"),
        transport=transport,
    )

    transcript = asyncio.run(transcriber.transcribe(AudioChunk(pcm=b"\0\0")))

    assert transcript.text == "check"
    assert len(transport.calls) == 1
    url, body, headers, timeout = transport.calls[0]
    assert url == "https://api.elevenlabs.io/v1/speech-to-text"
    assert timeout == 20.0
    assert headers["xi-api-key"] == "test-key"
    assert headers["Accept"] == "application/json"
    assert "multipart/form-data; boundary=" in headers["Content-Type"]

    assert b'name="model_id"\r\n\r\nscribe-custom' in body
    assert b'name="file_format"\r\n\r\npcm_s16le_16' in body
    assert b'name="language_code"\r\n\r\nen' in body
    assert b'name="keyterms"\r\n\r\ncall' in body
    assert b'name="file"; filename="speech.pcm"' in body
    assert b"Content-Type: application/octet-stream" in body
    assert b"RIFF" not in body


def test_elevenlabs_transcriber_extracts_text_from_tuple_wrapped_chunks():
    transport = _RecordingSpeechToTextTransport()
    transport.result = {"chunks": ({"text": ""}, {"text": "call"})}
    transcriber = ElevenLabsSpeechTranscriber(
        ElevenLabsSpeechTranscriptionConfig(api_key="test-key"),
        transport=transport,
    )

    transcript = asyncio.run(transcriber.transcribe(AudioChunk(pcm=b"\0\0")))

    assert transcript.text == "call"


def test_elevenlabs_stt_can_disable_language_field():
    transport = _RecordingSpeechToTextTransport()
    transcriber = ElevenLabsSpeechTranscriber(
        ElevenLabsSpeechTranscriptionConfig(api_key="test-key", language=None),
        transport=transport,
    )

    asyncio.run(transcriber.transcribe(AudioChunk(pcm=b"\0\0")))

    assert b'name="language_code"' not in transport.calls[0][1]


def test_pcm_payload_rejects_non_16_bit_pcm():
    with pytest.raises(VoiceRuntimeError, match="Expected 16-bit PCM"):
        _pcm_bytes(AudioChunk(pcm=b"\0", sample_width=1))


def test_elevenlabs_transcriber_loads_lazy_config_from_env(monkeypatch):
    monkeypatch.setenv(ELEVENLABS_API_KEY_ENV, "env-key")
    monkeypatch.setenv(ELEVENLABS_STT_MODEL_ENV, "scribe-env")
    transport = _RecordingSpeechToTextTransport()

    asyncio.run(ElevenLabsSpeechTranscriber(transport=transport).transcribe(AudioChunk(pcm=b"\0\0")))

    assert b'name="model_id"\r\n\r\nscribe-env' in transport.calls[0][1]
