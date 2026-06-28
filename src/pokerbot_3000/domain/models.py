"""Pydantic models for game state, actions, observations, and events."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pokerbot_3000.domain.cards import Card  # noqa: TC001


class Street(StrEnum):
    """Texas Hold'em street names."""

    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


class PlayerType(StrEnum):
    """Supported player/client kinds."""

    HUMAN = "human"
    ROBOT_AGENT = "robot_agent"
    WEB_AGENT = "web_agent"


class PlayerStatus(StrEnum):
    """Public player lifecycle state for a hand."""

    ACTIVE = "active"
    IN_HAND = "in_hand"
    FOLDED = "folded"
    ALL_IN = "all_in"
    OUT = "out"


class ActionType(StrEnum):
    """Poker action names accepted by the skeleton."""

    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE_TO = "raise_to"
    ALL_IN = "all_in"
    CONFIRM = "confirm"
    CANCEL = "cancel"


class EventType(StrEnum):
    """Event types stored by the in-memory orchestrator."""

    SYSTEM = "system"
    GAME_STARTED = "game_started"
    GAME_STOPPED = "game_stopped"
    ACTION_PROPOSED = "action_proposed"
    ACTION_COMMITTED = "action_committed"
    AGENT_DECISION = "agent_decision"
    ENGINE_PAUSED = "engine_paused"
    VISION_OBSERVATION = "vision_observation"
    PRIVATE_CARD_OBSERVATION = "private_card_observation"
    PRESENTATION_COMMAND = "presentation_command"


class ClientId(StrEnum):
    """Known orchestrator clients."""

    DASHBOARD = "dashboard"
    REACHY = "reachy"
    ELIZA = "eliza"
    VOICE = "voice"
    PUBLIC_VISION = "public_vision"


class ClientConnectionState(StrEnum):
    """Client connectivity state."""

    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    READY = "ready"
    ERROR = "error"


class PendingInputType(StrEnum):
    """External input currently blocking automated orchestration."""

    HUMAN_ACTION = "human_action"
    PRIVATE_CARDS = "private_cards"


class PokerBaseModel(BaseModel):
    """Base model with strict-ish defaults for API contracts."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class PokerAction(PokerBaseModel):
    """A proposed poker action."""

    type: ActionType
    amount: int | None = Field(default=None, ge=0)
    unit: str = "chips"

    @model_validator(mode="after")
    def validate_amount(self) -> PokerAction:
        """Require chip amounts only for amount-bearing actions."""
        if self.type in {ActionType.BET, ActionType.RAISE_TO} and self.amount is None:
            msg = f"{self.type} requires an amount."
            raise ValueError(msg)
        if self.type not in {ActionType.BET, ActionType.RAISE_TO, ActionType.CALL} and self.amount is not None:
            msg = f"{self.type} should not include an amount."
            raise ValueError(msg)
        return self


class PlayerState(PokerBaseModel):
    """Public state for one seat."""

    name: str
    type: PlayerType
    status: PlayerStatus
    stack: int = Field(ge=0)
    committed_this_street: int = Field(default=0, ge=0)


class PendingInput(PokerBaseModel):
    """External input the orchestrator needs before it can keep running."""

    type: PendingInputType
    seat: int = Field(ge=1, le=3)
    agent_id: str | None = None
    client_id: ClientId | None = None
    reason: str


class PublicGameState(PokerBaseModel):
    """Public game state that can be broadcast to all clients."""

    hand_id: str
    game_type: str = "texas_holdem"
    betting_structure: str = "no_limit"
    street: Street
    dealer_seat: int = Field(ge=1, le=3)
    active_player_seat: int = Field(ge=1, le=3)
    board: list[Card] = Field(default_factory=list, max_length=5)
    board_source: str = "manual"
    board_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    pot: int = Field(default=0, ge=0)
    current_bet_to_call: int = Field(default=0, ge=0)
    min_raise_to: int = Field(default=20, ge=0)
    players: dict[int, PlayerState]
    legal_actions: list[ActionType] = Field(default_factory=list)
    automation_status: str = "stopped"
    waiting_for: PendingInput | None = None
    last_actions: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    @field_validator("board")
    @classmethod
    def validate_board_progression(cls, board: list[Card]) -> list[Card]:
        """Keep board-card counts aligned with Hold'em streets."""
        if len(board) in {1, 2}:
            msg = "Board must have 0, 3, 4, or 5 cards."
            raise ValueError(msg)
        return board


class PrivateAgentState(PokerBaseModel):
    """Private state routed only to the owning agent."""

    agent_id: str
    seat: int = Field(ge=1, le=3)
    hole_cards: list[Card] = Field(default_factory=list, max_length=2)
    source: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_updated_at: datetime | None = None


class ClientStatus(PokerBaseModel):
    """Current status for an external client or bridge."""

    client_id: ClientId
    connection: ClientConnectionState
    status: str = "waiting"
    detail: str | None = None


class GameEvent(PokerBaseModel):
    """A replayable event-log item."""

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: EventType
    source: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PublicTableObservation(PokerBaseModel):
    """Structured public table perception result."""

    source: str = "main_table_camera"
    dealer_seat: int | None = Field(default=None, ge=1, le=3)
    board_cards: list[Card] = Field(default_factory=list, max_length=5)
    street_hint: Street | None = None
    pot_has_chips: bool | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = None


class PrivateCardObservation(PokerBaseModel):
    """Structured private card perception result for one agent."""

    agent_id: str
    seat: int = Field(ge=1, le=3)
    hole_cards: list[Card] = Field(max_length=2)
    source: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str | None = None


class HumanActionInput(PokerBaseModel):
    """Human action input accepted while the engine is paused for the human."""

    source: str = "voice"
    seat: int = Field(default=1, ge=1, le=3)
    action: PokerAction
    raw_transcript: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExternalInputResult(PokerBaseModel):
    """Result returned after the engine consumes an external input."""

    accepted: bool
    reason: str
    events: list[GameEvent]
    state: PublicGameState


class OperatorControlResult(PokerBaseModel):
    """Result returned after an operator starts or stops the game."""

    accepted: bool
    reason: str
    events: list[GameEvent]
    state: PublicGameState


class ObservationReceipt(PokerBaseModel):
    """Response body after recording an observation."""

    accepted: bool
    reason: str
    event: GameEvent
