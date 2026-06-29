"""Deterministic poker voice command grammar."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, cast

from pokerbot_3000.domain.models import ActionType, HumanActionInput, HumanTableTalkInput, PokerAction

if TYPE_CHECKING:
    from pokerbot_3000.ports.voice import VoiceTranscript

_ACTION_SOURCE: Final = "voice"
_HUMAN_SEAT: Final = 1
_THOUSAND: Final = 1000
_UNSAFE_CONTEXT = re.compile(
    r"\b(?:maybe|should i|could i|can i|what(?:'s| is)|how much|don't|do not|not|never|no)\b",
)
_PUNCTUATION = re.compile(r"[^a-z0-9\s]")
_DIGIT_AMOUNT = re.compile(r"\b\d[\d,]*\b")
_ACTION_PREFIX = re.compile(r"^(?:i|i am|i'm|im|i will|i'll|ill|let's|lets|please)\s+")

_SMALL_NUMBERS: Final = {
    "zero": 0,
    "oh": 0,
    "o": 0,
    "one": 1,
    "a": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS: Final = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES: Final = {"hundred": 100, "thousand": 1000, "grand": 1000, "k": 1000}
_NOISE_WORDS: Final = {"and", "chips", "chip", "dollars", "dollar", "please", "it", "to"}


@dataclass(frozen=True, slots=True)
class DeterministicVoiceCommandParser:
    """Parse clear poker table speech into human actions."""

    seat: int = _HUMAN_SEAT

    def parse(self, transcript: VoiceTranscript) -> HumanActionInput | HumanTableTalkInput | None:
        """Parse one transcript into a human action when it is unambiguous."""
        text = _normalize(transcript.text)
        if not text:
            return None

        table_talk = _parse_table_talk(text, transcript, seat=self.seat)
        if table_talk is not None:
            return table_talk

        if _UNSAFE_CONTEXT.search(text):
            return None

        action = _parse_action(text)
        if action is None:
            return None
        return HumanActionInput(
            source=_ACTION_SOURCE,
            seat=self.seat,
            action=action,
            raw_transcript=transcript.text,
            confidence=transcript.confidence,
        )


def _parse_table_talk(text: str, transcript: VoiceTranscript, *, seat: int) -> HumanTableTalkInput | None:
    matches = list(re.finditer(r"\b(?:reachy|eliza)\b", text))
    if not matches:
        return None
    target = min(matches, key=lambda match: match.start())
    message = f"{text[: target.start()]} {text[target.end() :]}".strip()
    message = re.sub(r"^(?:hey|hi|yo|ok|okay)\s+", "", message).strip()
    if not message or message in {"hey", "hi", "yo", "ok", "okay"}:
        return None
    target_agent_id = cast('Literal["reachy", "eliza"]', target.group(0))
    return HumanTableTalkInput(
        source=_ACTION_SOURCE,
        seat=seat,
        target_agent_id=target_agent_id,
        message=message,
        raw_transcript=transcript.text,
        confidence=transcript.confidence,
    )


def _parse_action(text: str) -> PokerAction | None:
    action: PokerAction | None = None
    if re.search(r"\ball\s+in\b", text):
        action = PokerAction(type=ActionType.ALL_IN)
    elif (amount := _amount_after(text, (r"\braise\s+to\s+", r"\braise\s+", r"\bmake\s+it\s+"))) is not None:
        action = PokerAction(type=ActionType.RAISE_TO, amount=amount)
    elif (amount := _amount_after(text, (r"\bbet\s+", r"\bwager\s+"))) is not None:
        action = PokerAction(type=ActionType.BET, amount=amount)
    elif (amount := _amount_after(text, (r"\bcall\s+",))) is not None:
        action = PokerAction(type=ActionType.CALL, amount=amount)
    else:
        stripped = _strip_prefix(text)
        if re.fullmatch(r"(?:fold|folding)", stripped):
            action = PokerAction(type=ActionType.FOLD)
        elif re.fullmatch(r"(?:check|checking|check it)", stripped):
            action = PokerAction(type=ActionType.CHECK)
        elif re.fullmatch(r"(?:call|calling|call it)", stripped):
            action = PokerAction(type=ActionType.CALL)
    return action


def _amount_after(text: str, prefixes: tuple[str, ...]) -> int | None:
    for prefix in prefixes:
        match = re.search(prefix, text)
        if match is None:
            continue
        amount = _parse_amount(text[match.end() :])
        if amount is not None:
            return amount
    return None


def _parse_amount(text: str) -> int | None:
    digit_match = _DIGIT_AMOUNT.search(text)
    if digit_match is not None:
        return int(digit_match.group(0).replace(",", ""))

    tokens = [token for token in text.split() if token not in _NOISE_WORDS]
    if not tokens:
        return None

    total = 0
    current = 0
    saw_number = False
    for token in tokens:
        if token in _SMALL_NUMBERS:
            current += _SMALL_NUMBERS[token]
            saw_number = True
            continue
        if token in _TENS:
            current += _TENS[token]
            saw_number = True
            continue
        if token in _SCALES and saw_number:
            scale = _SCALES[token]
            current = max(current, 1) * scale
            if scale >= _THOUSAND:
                total += current
                current = 0
            continue
        break

    amount = total + current
    return amount if saw_number and amount > 0 else None


def _normalize(text: str) -> str:
    normalized = re.sub(r"(?<=\d),(?=\d)", "", text.casefold()).replace("-", " ")
    normalized = _PUNCTUATION.sub(" ", normalized)
    return " ".join(normalized.split())


def _strip_prefix(text: str) -> str:
    previous = text
    while True:
        stripped = _ACTION_PREFIX.sub("", previous, count=1)
        if stripped == previous:
            return stripped
        previous = stripped
