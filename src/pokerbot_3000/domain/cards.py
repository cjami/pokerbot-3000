"""Playing card models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class CardRank(StrEnum):
    """Canonical card ranks."""

    ACE = "ace"
    KING = "king"
    QUEEN = "queen"
    JACK = "jack"
    TEN = "10"
    NINE = "9"
    EIGHT = "8"
    SEVEN = "7"
    SIX = "6"
    FIVE = "5"
    FOUR = "4"
    THREE = "3"
    TWO = "2"


class CardSuit(StrEnum):
    """Canonical card suits."""

    SPADES = "spades"
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"


class Card(BaseModel):
    """A friendly structured playing card."""

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=True)

    rank: CardRank
    suit: CardSuit

    @property
    def label(self) -> str:
        """Return a human display label."""
        return f"{str(self.rank).title()} of {str(self.suit).title()}"
