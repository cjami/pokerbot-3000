"""Cerebras-backed LLM gateway for Gemma poker tasks."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from pydantic import TypeAdapter

from pokerbot_3000.domain.cards import Card, CardRank, CardSuit
from pokerbot_3000.domain.models import ActionType, PokerAction, PrivateCardObservation, PublicTableObservation, Street
from pokerbot_3000.llm.prompt_catalog import PromptCatalog, load_prompts
from pokerbot_3000.ports.llm import AgentBanterDecision, AgentDecision, ImageFrame

if TYPE_CHECKING:
    from pathlib import Path

    from pokerbot_3000.domain.models import GameEvent, HumanTableTalkInput, PrivateAgentState, PublicGameState

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]
type JsonTransport = Callable[[str, JsonObject, Mapping[str, str], float], JsonObject]

CEREBRAS_API_KEY_ENV: Final = "CEREBRAS_API_KEY"
CEREBRAS_MODEL_ENV: Final = "CEREBRAS_MODEL"
CEREBRAS_BASE_URL_ENV: Final = "CEREBRAS_BASE_URL"
CEREBRAS_TIMEOUT_ENV: Final = "CEREBRAS_TIMEOUT_SECONDS"
DEFAULT_CEREBRAS_MODEL: Final = "gemma-4-31b"
DEFAULT_CEREBRAS_BASE_URL: Final = "https://api.cerebras.ai/v1/"
DEFAULT_TIMEOUT_SECONDS: Final = 30.0

_CARD_LIST_ADAPTER = TypeAdapter(list[Card])
_NULLABLE_STRING: Final[JsonObject] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
_ACTION_TYPE_VALUES: Final = [action.value for action in ActionType]
_CARD_RANK_VALUES: Final = [rank.value for rank in CardRank]
_CARD_SUIT_VALUES: Final = [suit.value for suit in CardSuit]
_STREET_VALUES: Final = [street.value for street in Street]
_AGENT_EMOTION_VALUES: Final = ["calm", "confident", "celebrate", "confused", "sad"]


class CerebrasConfigurationError(RuntimeError):
    """Raised when Cerebras configuration is missing or invalid."""


class CerebrasClientError(RuntimeError):
    """Raised when Cerebras returns an unusable response."""


@dataclass(frozen=True, slots=True)
class CerebrasConfig:
    """Runtime configuration for Cerebras inference."""

    api_key: str
    model: str = DEFAULT_CEREBRAS_MODEL
    base_url: str = DEFAULT_CEREBRAS_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> CerebrasConfig:
        """Load Cerebras settings from environment variables and an optional dotenv file."""
        load_dotenv(dotenv_path=env_file)

        api_key = os.getenv(CEREBRAS_API_KEY_ENV)
        if not api_key:
            msg = f"Set {CEREBRAS_API_KEY_ENV} in your environment or .env file."
            raise CerebrasConfigurationError(msg)

        return cls(
            api_key=api_key,
            model=os.getenv(CEREBRAS_MODEL_ENV, DEFAULT_CEREBRAS_MODEL),
            base_url=_normalized_base_url(os.getenv(CEREBRAS_BASE_URL_ENV, DEFAULT_CEREBRAS_BASE_URL)),
            timeout_seconds=_env_float(CEREBRAS_TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS),
        )


@dataclass(frozen=True, slots=True)
class CerebrasAccessCheck:
    """Result from a lightweight Cerebras connectivity check."""

    ok: bool
    model: str
    model_listed: bool
    reply: str


class CerebrasLlmClient:
    """OpenAI-compatible Cerebras client implementing poker LLM tasks."""

    def __init__(
        self,
        config: CerebrasConfig,
        transport: JsonTransport | None = None,
        prompts: PromptCatalog | None = None,
    ) -> None:
        """Create a client with injectable transport for tests."""
        self._config = config
        self._transport = transport or _urllib_json_transport
        self._prompts = prompts or load_prompts()

    async def list_model_ids(self) -> list[str]:
        """Return model IDs visible to the configured Cerebras account."""
        response = await self._request_json("models", {})
        data = response.get("data")
        if not isinstance(data, list):
            msg = "Cerebras models response did not contain a data list."
            raise CerebrasClientError(msg)
        return [model_id for model in data if (model_id := _model_id(model)) is not None]

    async def check_access(self) -> CerebrasAccessCheck:
        """Verify that the configured model is listed and can complete a tiny request."""
        model_ids = await self.list_model_ids()
        model_listed = self._config.model in model_ids
        reply = await self._chat_text(
            [
                {
                    "role": "user",
                    "content": "Reply with exactly: pokerbot-ok",
                },
            ],
            max_tokens=8,
            temperature=0.0,
        )
        return CerebrasAccessCheck(
            ok=model_listed and "pokerbot-ok" in reply.lower(),
            model=self._config.model,
            model_listed=model_listed,
            reply=reply.strip(),
        )

    async def read_public_table(self, frame: ImageFrame) -> PublicTableObservation:
        """Read labelled public table state from a frame."""
        payload = await self._chat_json(
            [
                _system_message(self._prompts.read_public_table_system),
                _image_user_message(frame, self._prompts.read_public_table_user),
            ],
            max_tokens=300,
            response_format=PUBLIC_TABLE_RESPONSE_FORMAT,
        )
        payload["source"] = frame.source
        return PublicTableObservation.model_validate(payload)

    async def read_board_cards(self, frame_or_crop: ImageFrame) -> list[Card]:
        """Read visible cards from the public board zone."""
        payload = await self._chat_json(
            [
                _system_message(self._prompts.read_board_cards_system),
                _image_user_message(frame_or_crop, self._prompts.read_board_cards_user),
            ],
            max_tokens=120,
            response_format=BOARD_CARDS_RESPONSE_FORMAT,
        )
        cards = payload.get("cards", [])
        return _CARD_LIST_ADAPTER.validate_python(cards)

    async def read_hole_cards(self, agent_id: str, frame: ImageFrame) -> PrivateCardObservation:
        """Read private hole cards for one agent."""
        seat = _seat_for_agent(agent_id)
        payload = await self._chat_json(
            [
                _system_message(self._prompts.read_hole_cards_system),
                _image_user_message(frame, self._prompts.read_hole_cards_user),
            ],
            max_tokens=220,
            response_format=HOLE_CARDS_RESPONSE_FORMAT,
        )
        payload["agent_id"] = agent_id
        payload["seat"] = seat
        payload.setdefault("source", frame.source)
        return PrivateCardObservation.model_validate(payload)

    async def read_revealed_cards(self, frame: ImageFrame) -> list[Card]:
        """Read two revealed hole cards from one seat crop."""
        payload = await self._chat_json(
            [
                _system_message(self._prompts.read_hole_cards_system),
                _image_user_message(frame, self._prompts.read_hole_cards_user),
            ],
            max_tokens=220,
            response_format=HOLE_CARDS_RESPONSE_FORMAT,
        )
        cards = payload.get("hole_cards", [])
        return _CARD_LIST_ADAPTER.validate_python(cards)

    async def decide_agent_action(
        self,
        agent_id: str,
        public_state: PublicGameState,
        private_state: PrivateAgentState,
    ) -> AgentDecision:
        """Choose an agent action from public and private state."""
        payload = await self._chat_json(
            [
                _system_message(self._prompts.decide_agent_action_system),
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent_id": agent_id,
                            "public_state": public_state.model_dump(mode="json"),
                            "private_state": private_state.model_dump(mode="json"),
                        },
                    ),
                },
            ],
            max_tokens=360,
            response_format=AGENT_DECISION_RESPONSE_FORMAT,
        )
        return AgentDecision(
            agent_id=agent_id,
            action=PokerAction.model_validate(payload.get("action")),
            speech=cast("str | None", payload.get("speech")),
            reaction=_object_dict(payload.get("reaction"), default={"intent": "announce_action"}),
            confidence=_confidence(payload.get("confidence")),
        )

    async def generate_table_talk(self, agent_id: str, event: GameEvent) -> str:
        """Generate a table-talk line for an agent."""
        text = await self._chat_text(
            [
                _system_message(self._prompts.generate_table_talk_system),
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent_id": agent_id,
                            "event": event.model_dump(mode="json"),
                            "instruction": "Return one line, 16 words or fewer.",
                        },
                    ),
                },
            ],
            max_tokens=60,
            temperature=0.7,
        )
        return text.strip().strip('"')

    async def respond_to_human_table_talk(
        self,
        request: HumanTableTalkInput,
        public_state: PublicGameState,
    ) -> AgentBanterDecision:
        """Respond to direct human speech addressed to one agent."""
        payload = await self._chat_json(
            [
                _system_message(self._prompts.agent_banter_system),
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "direct_reply",
                            "target_agent_id": request.target_agent_id,
                            "human_message": request.message,
                            "public_state": public_state.model_dump(mode="json"),
                            "instruction": "Return speech from the addressed agent.",
                        },
                    ),
                },
            ],
            max_tokens=160,
            response_format=AGENT_BANTER_RESPONSE_FORMAT,
        )
        return _agent_banter_decision(payload, default_agent_id=request.target_agent_id)

    async def react_to_human_action(
        self,
        event: GameEvent,
        public_state: PublicGameState,
    ) -> AgentBanterDecision:
        """Optionally react to one committed human poker action."""
        payload = await self._chat_json(
            [
                _system_message(self._prompts.agent_banter_system),
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "task": "human_action_reaction",
                            "event": event.model_dump(mode="json"),
                            "public_state": public_state.model_dump(mode="json"),
                            "instruction": "Choose reachy, eliza, or no reaction.",
                        },
                    ),
                },
            ],
            max_tokens=160,
            response_format=AGENT_BANTER_RESPONSE_FORMAT,
        )
        return _agent_banter_decision(payload, default_agent_id=None)

    async def _chat_json(
        self,
        messages: list[JsonObject],
        *,
        max_tokens: int,
        response_format: JsonObject,
    ) -> JsonObject:
        text = await self._chat_text(
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format=response_format,
        )
        try:
            parsed = json.loads(_extract_json_text(text))
        except json.JSONDecodeError as exc:
            msg = "Cerebras response did not contain valid JSON."
            raise CerebrasClientError(msg) from exc
        if not isinstance(parsed, dict):
            msg = "Cerebras JSON response was not an object."
            raise CerebrasClientError(msg)
        return cast("JsonObject", parsed)

    async def _chat_text(
        self,
        messages: list[JsonObject],
        *,
        max_tokens: int,
        temperature: float,
        response_format: JsonObject | None = None,
    ) -> str:
        payload: JsonObject = {
            "model": self._config.model,
            "messages": cast("JsonValue", messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if _messages_include_image(messages):
            payload["reasoning_effort"] = "none"
        response = await self._request_json("chat/completions", payload)
        return _first_message_text(response)

    async def _request_json(self, path: str, payload: JsonObject) -> JsonObject:
        url = urljoin(self._config.base_url, path)
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "pokerbot-3000/0.1",
        }
        return await asyncio.to_thread(self._transport, url, payload, headers, self._config.timeout_seconds)


def _urllib_json_transport(url: str, payload: JsonObject, headers: Mapping[str, str], timeout: float) -> JsonObject:
    data = None if not payload else json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=data, headers=dict(headers), method="GET" if data is None else "POST")  # noqa: S310
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        msg = f"Cerebras API request failed with HTTP {exc.code}: {_trim(detail)}"
        raise CerebrasClientError(msg) from exc
    except URLError as exc:
        msg = f"Could not reach Cerebras API: {exc.reason}"
        raise CerebrasClientError(msg) from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = "Cerebras API returned invalid JSON."
        raise CerebrasClientError(msg) from exc
    if not isinstance(parsed, dict):
        msg = "Cerebras API returned a non-object JSON response."
        raise CerebrasClientError(msg)
    return cast("JsonObject", parsed)


def _strict_response_format(name: str, schema: JsonObject) -> JsonObject:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _object_schema(properties: JsonObject, required: list[str]) -> JsonObject:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": cast("JsonValue", required),
    }


def _array_schema(item_schema: JsonObject) -> JsonObject:
    return {"type": "array", "items": item_schema}


def _nullable(schema: JsonObject) -> JsonObject:
    return {"anyOf": [schema, {"type": "null"}]}


def _string_enum(values: Sequence[str]) -> JsonObject:
    return {"type": "string", "enum": cast("JsonValue", values)}


def _integer_enum(values: Sequence[int]) -> JsonObject:
    return {"type": "integer", "enum": cast("JsonValue", values)}


CARD_SCHEMA: Final = _object_schema(
    {
        "rank": _string_enum(_CARD_RANK_VALUES),
        "suit": _string_enum(_CARD_SUIT_VALUES),
    },
    ["rank", "suit"],
)
CONFIDENCE_SCHEMA: Final[JsonObject] = {"type": "number", "minimum": 0, "maximum": 1}

PUBLIC_TABLE_RESPONSE_FORMAT: Final = _strict_response_format(
    "public_table_observation",
    _object_schema(
        {
            "dealer_seat": _nullable(_integer_enum([1, 2, 3])),
            "board_cards": _array_schema(CARD_SCHEMA),
            "street_hint": _nullable(_string_enum(_STREET_VALUES)),
            "pot_has_chips": _nullable({"type": "boolean"}),
            "confidence": CONFIDENCE_SCHEMA,
            "notes": _NULLABLE_STRING,
        },
        ["dealer_seat", "board_cards", "street_hint", "pot_has_chips", "confidence", "notes"],
    ),
)
BOARD_CARDS_RESPONSE_FORMAT: Final = _strict_response_format(
    "board_cards",
    _object_schema(
        {"cards": _array_schema(CARD_SCHEMA)},
        ["cards"],
    ),
)
HOLE_CARDS_RESPONSE_FORMAT: Final = _strict_response_format(
    "hole_cards",
    _object_schema(
        {
            "hole_cards": _array_schema(CARD_SCHEMA),
            "confidence": CONFIDENCE_SCHEMA,
            "notes": _NULLABLE_STRING,
        },
        ["hole_cards", "confidence", "notes"],
    ),
)
AGENT_DECISION_RESPONSE_FORMAT: Final = _strict_response_format(
    "agent_decision",
    _object_schema(
        {
            "action": _object_schema(
                {
                    "type": _string_enum(_ACTION_TYPE_VALUES),
                    "amount": _nullable({"type": "integer", "minimum": 0}),
                },
                ["type", "amount"],
            ),
            "speech": _NULLABLE_STRING,
            "reaction": _object_schema({"intent": {"type": "string"}}, ["intent"]),
            "confidence": CONFIDENCE_SCHEMA,
        },
        ["action", "speech", "reaction", "confidence"],
    ),
)
AGENT_BANTER_RESPONSE_FORMAT: Final = _strict_response_format(
    "agent_banter",
    _object_schema(
        {
            "agent_id": _nullable(_string_enum(["reachy", "eliza"])),
            "speech": _NULLABLE_STRING,
            "reaction": _object_schema({"intent": {"type": "string"}}, ["intent"]),
            "emotion": _string_enum(_AGENT_EMOTION_VALUES),
            "confidence": CONFIDENCE_SCHEMA,
        },
        ["agent_id", "speech", "reaction", "emotion", "confidence"],
    ),
)


def _system_message(content: str) -> JsonObject:
    return {"role": "system", "content": content}


def _image_user_message(frame: ImageFrame, text: str) -> JsonObject:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": frame.data_uri}},
        ],
    }


def _first_message_text(response: JsonObject) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        msg = "Cerebras chat response did not contain choices."
        raise CerebrasClientError(msg)
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        msg = "Cerebras chat choice was not an object."
        raise CerebrasClientError(msg)
    message = first_choice.get("message")
    if not isinstance(message, dict):
        msg = "Cerebras chat choice did not contain a message object."
        raise CerebrasClientError(msg)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [text for part in content if (text := _text_part(part)) is not None]
        return "\n".join(text_parts)
    msg = "Cerebras chat message did not contain text content."
    raise CerebrasClientError(msg)


def _messages_include_image(messages: list[JsonObject]) -> bool:
    return any(_content_includes_image(message.get("content")) for message in messages)


def _content_includes_image(content: object) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index != -1]
    if not starts:
        msg = "Cerebras response did not include a JSON object."
        raise CerebrasClientError(msg)
    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end < start:
        msg = "Cerebras response did not include complete JSON."
        raise CerebrasClientError(msg)
    return stripped[start : end + 1]


def _model_id(value: object) -> str | None:
    if isinstance(value, dict) and isinstance(model_id := value.get("id"), str):
        return model_id
    return None


def _text_part(value: object) -> str | None:
    if isinstance(value, dict) and isinstance(text := value.get("text"), str):
        return text
    return None


def _seat_for_agent(agent_id: str) -> int:
    seats = {"reachy": 2, "eliza": 3}
    try:
        return seats[agent_id]
    except KeyError as exc:
        msg = f"Unknown poker agent {agent_id!r}."
        raise ValueError(msg) from exc


def _object_dict(value: object, *, default: dict[str, object]) -> dict[str, object]:
    if value is None:
        return default
    if not isinstance(value, dict):
        msg = "Expected object for reaction."
        raise CerebrasClientError(msg)
    if not all(isinstance(key, str) for key in value):
        msg = "Expected string keys for reaction."
        raise CerebrasClientError(msg)
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _agent_banter_decision(payload: JsonObject, *, default_agent_id: str | None) -> AgentBanterDecision:
    agent_id = payload.get("agent_id")
    if agent_id is None:
        agent_id = default_agent_id
    if agent_id is not None and agent_id not in {"reachy", "eliza"}:
        msg = f"Unknown poker agent {agent_id!r}."
        raise CerebrasClientError(msg)
    speech = payload.get("speech")
    if speech is not None and not isinstance(speech, str):
        msg = "Expected string or null speech from Cerebras response."
        raise CerebrasClientError(msg)
    return AgentBanterDecision(
        agent_id=cast("str | None", agent_id),
        speech=speech.strip() if isinstance(speech, str) and speech.strip() else None,
        reaction=_object_dict(payload.get("reaction"), default={"intent": "table_talk"}),
        confidence=_confidence(payload.get("confidence")),
        emotion=_agent_emotion(payload.get("emotion")),
    )


def _agent_emotion(value: object) -> str:
    if value is None:
        return "calm"
    if not isinstance(value, str) or value not in _AGENT_EMOTION_VALUES:
        msg = "Expected preset agent emotion from Cerebras response."
        raise CerebrasClientError(msg)
    return value


def _confidence(value: object) -> float:
    if not isinstance(value, int | float | str):
        msg = "Expected numeric confidence from Cerebras response."
        raise CerebrasClientError(msg)
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        msg = "Expected numeric confidence from Cerebras response."
        raise CerebrasClientError(msg) from exc
    return max(0.0, min(1.0, confidence))


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
        raise CerebrasConfigurationError(msg) from exc


def _trim(value: str, max_length: int = 500) -> str:
    compact = " ".join(value.split())
    if len(compact) <= max_length:
        return compact
    return f"{compact[:max_length]}..."
