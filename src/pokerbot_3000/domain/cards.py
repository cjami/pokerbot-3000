"""Card notation helpers."""

from __future__ import annotations

from typing import Annotated

from pydantic import StringConstraints

CARD_PATTERN = r"^[AKQJT98765432][shdc]$"
type Card = Annotated[str, StringConstraints(pattern=CARD_PATTERN)]
