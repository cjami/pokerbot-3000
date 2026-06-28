"""Cached Markdown prompt loading for LLM tasks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Final

PROMPT_PACKAGE: Final = "pokerbot_3000.llm.prompts"


@dataclass(frozen=True, slots=True)
class PromptCatalog:
    """Markdown prompts loaded once for the process."""

    read_public_table_system: str
    read_public_table_user: str
    read_board_cards_system: str
    read_board_cards_user: str
    read_hole_cards_system: str
    read_hole_cards_user: str
    decide_agent_action_system: str
    generate_table_talk_system: str


@cache
def load_prompts() -> PromptCatalog:
    """Load all prompt files once and reuse them for the process lifetime."""
    return PromptCatalog(
        read_public_table_system=_read_prompt("read_public_table.system.md"),
        read_public_table_user=_read_prompt("read_public_table.user.md"),
        read_board_cards_system=_read_prompt("read_board_cards.system.md"),
        read_board_cards_user=_read_prompt("read_board_cards.user.md"),
        read_hole_cards_system=_read_prompt("read_hole_cards.system.md"),
        read_hole_cards_user=_read_prompt("read_hole_cards.user.md"),
        decide_agent_action_system=_read_prompt("decide_agent_action.system.md"),
        generate_table_talk_system=_read_prompt("generate_table_talk.system.md"),
    )


def _read_prompt(name: str) -> str:
    return files(PROMPT_PACKAGE).joinpath(name).read_text(encoding="utf-8").strip()
