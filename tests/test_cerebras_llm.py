import asyncio
import json
from collections.abc import Mapping
from typing import Any, cast

import pytest

from pokerbot_3000.llm.cerebras import (
    CEREBRAS_API_KEY_ENV,
    CEREBRAS_MODEL_ENV,
    CerebrasConfig,
    CerebrasConfigurationError,
    CerebrasLlmClient,
    JsonObject,
)
from pokerbot_3000.llm.prompt_catalog import load_prompts
from pokerbot_3000.ports.llm import ImageFrame


def test_cerebras_config_loads_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv(CEREBRAS_API_KEY_ENV, raising=False)
    monkeypatch.delenv(CEREBRAS_MODEL_ENV, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("CEREBRAS_API_KEY=test-key\nCEREBRAS_MODEL=test-model\n", encoding="utf-8")

    config = CerebrasConfig.from_env(env_file)

    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.base_url == "https://api.cerebras.ai/v1/"


def test_cerebras_config_requires_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv(CEREBRAS_API_KEY_ENV, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("CEREBRAS_MODEL=test-model\n", encoding="utf-8")

    with pytest.raises(CerebrasConfigurationError, match=CEREBRAS_API_KEY_ENV):
        CerebrasConfig.from_env(env_file)


def test_cerebras_access_check_uses_models_and_chat_endpoints():
    calls: list[str] = []

    def fake_transport(url: str, _payload: JsonObject, _headers: Mapping[str, str], _timeout: float) -> JsonObject:
        calls.append(url)
        if url.endswith("/models"):
            return {"data": [{"id": "gemma-4-31b"}]}
        return {"choices": [{"message": {"content": "pokerbot-ok"}}]}

    client = CerebrasLlmClient(CerebrasConfig(api_key="test-key"), transport=fake_transport)

    result = asyncio.run(client.check_access())

    assert result.ok is True
    assert result.model_listed is True
    assert calls == [
        "https://api.cerebras.ai/v1/models",
        "https://api.cerebras.ai/v1/chat/completions",
    ]


def test_prompt_catalog_is_cached():
    assert load_prompts() is load_prompts()


def test_read_hole_cards_uses_strict_schema_and_python_metadata():
    captured_payloads: list[JsonObject] = []

    def fake_transport(url: str, payload: JsonObject, _headers: Mapping[str, str], _timeout: float) -> JsonObject:
        captured_payloads.append(payload)
        assert url.endswith("/chat/completions")
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "hole_cards": [
                                    {"rank": "ace", "suit": "spades"},
                                    {"rank": "king", "suit": "diamonds"},
                                ],
                                "confidence": 0.93,
                                "notes": None,
                            }
                        )
                    }
                }
            ]
        }

    client = CerebrasLlmClient(CerebrasConfig(api_key="test-key"), transport=fake_transport)
    frame = ImageFrame(source="reachy_camera", data_uri="data:image/png;base64,test")

    result = asyncio.run(client.read_hole_cards("reachy", frame))

    payload = captured_payloads[0]
    response_format = payload["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    json_schema = cast("dict[str, Any]", response_format["json_schema"])
    assert json_schema["strict"] is True
    schema = cast("dict[str, Any]", json_schema["schema"])
    properties = cast("dict[str, Any]", schema["properties"])
    card_schema = properties["hole_cards"]["items"]
    assert card_schema["properties"]["rank"]["enum"] == [
        "ace",
        "king",
        "queen",
        "jack",
        "10",
        "9",
        "8",
        "7",
        "6",
        "5",
        "4",
        "3",
        "2",
    ]
    assert card_schema["properties"]["suit"]["enum"] == ["spades", "hearts", "diamonds", "clubs"]
    assert "reachy" not in json.dumps(payload)
    assert "seat" not in json.dumps(payload["messages"])
    assert result.agent_id == "reachy"
    assert result.seat == 2
    assert result.source == "reachy_camera"
    assert [card.model_dump() for card in result.hole_cards] == [
        {"rank": "ace", "suit": "spades"},
        {"rank": "king", "suit": "diamonds"},
    ]
