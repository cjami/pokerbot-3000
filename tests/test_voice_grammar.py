from pokerbot_3000.domain.models import ActionType, HumanActionInput, HumanTableTalkInput
from pokerbot_3000.ports.voice import VoiceTranscript
from pokerbot_3000.voice import DeterministicVoiceCommandParser


def test_voice_grammar_parses_natural_table_actions():
    parser = DeterministicVoiceCommandParser()

    assert _parse_required(parser, "I fold").action.type == ActionType.FOLD
    assert _parse_required(parser, "check it").action.type == ActionType.CHECK
    assert _parse_required(parser, "I'm all in").action.type == ActionType.ALL_IN
    assert _parse_required(parser, "call").action.type == ActionType.CALL


def test_voice_grammar_parses_amount_actions_from_words_and_digits():
    parser = DeterministicVoiceCommandParser()

    bet = _parse_required(parser, "bet five hundred")
    raise_to = _parse_required(parser, "make it 1,200 chips")
    call = _parse_required(parser, "call twenty five")

    assert bet.action.type == ActionType.BET
    assert bet.action.amount == 500
    assert raise_to.action.type == ActionType.RAISE_TO
    assert raise_to.action.amount == 1200
    assert call.action.type == ActionType.CALL
    assert call.action.amount == 25


def test_voice_grammar_parses_common_asr_action_variants():
    parser = DeterministicVoiceCommandParser()

    assert _parse_required(parser, "raised it to two hundred").action.type == ActionType.RAISE_TO
    assert _parse_required(parser, "raised it to two hundred").action.amount == 200
    assert _parse_required(parser, "called").action.type == ActionType.CALL
    assert _parse_required(parser, "folded").action.type == ActionType.FOLD


def test_voice_grammar_rejects_unclear_or_unsafe_speech():
    parser = DeterministicVoiceCommandParser()

    assert parser.parse(_transcript("maybe I should call")) is None
    assert parser.parse(_transcript("five hundred")) is None
    assert parser.parse(_transcript("do not fold")) is None


def test_voice_grammar_parses_direct_agent_address():
    parser = DeterministicVoiceCommandParser()

    request = parser.parse(_transcript("Reachy, what do you make of this board?"))

    assert isinstance(request, HumanTableTalkInput)
    assert request.target_agent_id == "reachy"
    assert request.message == "what do you make of this board"


def test_voice_grammar_rejects_empty_agent_address():
    parser = DeterministicVoiceCommandParser()

    assert parser.parse(_transcript("hey Eliza")) is None


def test_voice_grammar_treats_named_action_words_as_banter_first():
    parser = DeterministicVoiceCommandParser()

    request = parser.parse(_transcript("Eliza, I call your bluff"))

    assert isinstance(request, HumanTableTalkInput)
    assert request.target_agent_id == "eliza"
    assert request.message == "i call your bluff"


def _transcript(text: str) -> VoiceTranscript:
    return VoiceTranscript(text=text, confidence=0.91)


def _parse_required(parser: DeterministicVoiceCommandParser, text: str) -> HumanActionInput:
    request = parser.parse(_transcript(text))
    assert isinstance(request, HumanActionInput)
    return request
