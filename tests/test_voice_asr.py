import asyncio
import types

import pytest

from pokerbot_3000.ports.voice import AudioChunk
from pokerbot_3000.voice import (
    DEFAULT_PARAKEET_MODEL,
    LEGACY_UNIFIED_PARAKEET_MODEL,
    POKERBOT_VOICE_MODEL_ENV,
    ParakeetConfig,
    ParakeetSpeechTranscriber,
    VoiceRuntimeError,
)


class _FakeAsrModel:
    def __init__(self) -> None:
        self.model_names: list[str] = []

    def from_pretrained(self, *, model_name: str) -> "_LoadedModel":
        self.model_names.append(model_name)
        return _LoadedModel()


class _LoadedModel:
    def transcribe(self, paths: list[str]) -> list[str]:
        assert len(paths) == 1
        return ["check"]


def test_parakeet_config_uses_registered_model_by_default(monkeypatch):
    monkeypatch.delenv(POKERBOT_VOICE_MODEL_ENV, raising=False)

    assert ParakeetConfig.from_env().model_name == DEFAULT_PARAKEET_MODEL


def test_parakeet_config_maps_old_unified_model_to_registered_model(monkeypatch):
    monkeypatch.setenv(POKERBOT_VOICE_MODEL_ENV, LEGACY_UNIFIED_PARAKEET_MODEL)

    assert ParakeetConfig.from_env().model_name == DEFAULT_PARAKEET_MODEL


def test_parakeet_loader_passes_configured_model_to_nemo(monkeypatch):
    fake_model = _FakeAsrModel()
    fake_nemo = types.SimpleNamespace(models=types.SimpleNamespace(ASRModel=fake_model))
    monkeypatch.setattr("pokerbot_3000.voice.asr.importlib.import_module", lambda _name: fake_nemo)

    transcript = asyncio.run(
        ParakeetSpeechTranscriber(ParakeetConfig(model_name="custom-model")).transcribe(AudioChunk(pcm=b"\0\0"))
    )

    assert transcript.text == "check"
    assert fake_model.model_names == ["custom-model"]


def test_parakeet_loader_explains_conformer_config_mismatch(monkeypatch):
    class BrokenAsrModel:
        def from_pretrained(self, *, model_name: str) -> object:
            _ = model_name
            msg = "ConformerEncoder.__init__() got an unexpected keyword argument 'att_chunk_context_size'"
            raise TypeError(msg)

    fake_nemo = types.SimpleNamespace(models=types.SimpleNamespace(ASRModel=BrokenAsrModel()))
    monkeypatch.setattr("pokerbot_3000.voice.asr.importlib.import_module", lambda _name: fake_nemo)

    with pytest.raises(VoiceRuntimeError) as exc_info:
        asyncio.run(ParakeetSpeechTranscriber(ParakeetConfig(model_name="newer-model")).transcribe(AudioChunk(pcm=b"")))

    message = str(exc_info.value)
    assert "newer-model" in message
    assert f"{POKERBOT_VOICE_MODEL_ENV}={DEFAULT_PARAKEET_MODEL}" in message
