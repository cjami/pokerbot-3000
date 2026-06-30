import asyncio
import time
from typing import Any

from fastapi.testclient import TestClient

from pokerbot_3000.app.runtime import (
    AgentActionProcessor,
    AgentBanterSource,
    DashboardEventBroadcaster,
    DashboardRuntime,
    PrivateCardsFrameProcessor,
    PublicBoardFrameProcessor,
    RevealedCardsFrameProcessor,
)
from pokerbot_3000.app.server import create_app
from pokerbot_3000.domain.cards import Card
from pokerbot_3000.domain.models import (
    ClientId,
    GameEvent,
    HumanActionInput,
    HumanTableTalkInput,
    PendingInputType,
    PokerAction,
    PrivateAgentState,
    PrivateCardObservation,
    PublicGameState,
    PublicTableObservation,
    Street,
)
from pokerbot_3000.orchestrator import InMemoryOrchestrator
from pokerbot_3000.ports.llm import AgentBanterDecision, AgentDecision, ImageFrame
from pokerbot_3000.ports.perception import PrivateCardSource, PublicVisionSource
from pokerbot_3000.voice import BrowserAudioInput


class NoopPublicVision:
    """Public vision fake that reads browser-submitted frames."""

    @property
    def latest_frame(self) -> ImageFrame | None:
        """Return the fake latest frame."""
        return None

    async def observe_frame(self, frame: ImageFrame) -> PublicTableObservation:
        """Return an empty observation when a submitted frame is processed."""
        _ = frame
        return PublicTableObservation(confidence=0.0)


class StaticBoardVision:
    """Public vision fake that returns the same board for every submitted frame."""

    def __init__(self, cards: list[Card]) -> None:
        """Initialize the fixed card output."""
        self.cards = cards
        self.frames: list[ImageFrame] = []

    async def observe_frame(self, frame: ImageFrame) -> PublicTableObservation:
        """Return a fixed, high-confidence board observation."""
        self.frames.append(frame)
        return PublicTableObservation(
            source=frame.source,
            board_cards=self.cards,
            street_hint=Street.FLOP,
            confidence=0.9,
        )


class StaticRevealedCardsVision:
    """Showdown reveal fake that returns configured cards."""

    def __init__(self, cards: list[Card] | None = None) -> None:
        """Initialize the fixed revealed-card output."""
        self.cards = cards or [_card("9", "clubs"), _card("9", "diamonds")]
        self.frames: list[ImageFrame] = []

    async def read_revealed_cards(self, frame: ImageFrame) -> list[Card]:
        """Return fixed revealed cards for one crop."""
        self.frames.append(frame)
        return self.cards


class StaticPrivateCardsVision:
    """Private-card fake that returns configured cards."""

    def __init__(self, cards: list[Card] | None = None) -> None:
        """Initialize the fixed private-card output."""
        self.cards = cards if cards is not None else [_card("9", "clubs"), _card("9", "diamonds")]
        self.frames: list[ImageFrame] = []

    async def read_private_cards(self, agent_id: str, frame: ImageFrame) -> PrivateCardObservation:
        """Return fixed private cards for one client frame."""
        self.frames.append(frame)
        return PrivateCardObservation(
            agent_id=agent_id,
            seat={"reachy": 2, "eliza": 3}[agent_id],
            hole_cards=self.cards,
            source=frame.source,
            confidence=0.9,
        )


class FakeVoiceClient:
    """Voice fake that returns deterministic bytes."""

    async def synthesize_orchestrator(self, text: str) -> bytes:
        """Return fake MPEG bytes for a speech line."""
        return f"audio:{text}".encode()

    async def synthesize_eliza(self, text: str) -> bytes:
        """Return fake MPEG bytes for an Eliza speech line."""
        return f"eliza:{text}".encode()

    async def synthesize_reachy(self, text: str) -> bytes:
        """Return fake MPEG bytes for a Reachy speech line."""
        return f"reachy:{text}".encode()


class RecordingVoiceClient:
    """Voice fake that records synthesis requests."""

    def __init__(self) -> None:
        """Initialize the request log."""
        self.calls: list[str] = []

    async def synthesize_orchestrator(self, text: str) -> bytes:
        """Record the requested speech and return fake MPEG bytes."""
        self.calls.append(text)
        return f"audio:{text}".encode()

    async def synthesize_eliza(self, text: str) -> bytes:
        """Record the requested Eliza speech and return fake MPEG bytes."""
        self.calls.append(f"eliza:{text}")
        return f"eliza:{text}".encode()

    async def synthesize_reachy(self, text: str) -> bytes:
        """Record the requested Reachy speech and return fake MPEG bytes."""
        self.calls.append(f"reachy:{text}")
        return f"reachy:{text}".encode()


class FakeAgentDecisions:
    """Agent-decision fake that picks a simple legal action."""

    async def decide_agent_action(
        self,
        agent_id: str,
        public_state: PublicGameState,
        private_state: PrivateAgentState,
    ) -> AgentDecision:
        """Return a legal deterministic decision for app tests."""
        _ = private_state
        if "call" in public_state.legal_actions:
            action = PokerAction.model_validate({"type": "call"})
        elif "check" in public_state.legal_actions:
            action = PokerAction.model_validate({"type": "check"})
        else:
            action = PokerAction.model_validate({"type": "fold"})
        return AgentDecision(
            agent_id=agent_id,
            action=action,
            speech=f"{agent_id} {action.type}",
            reaction={"intent": "announce_action"},
            confidence=0.9,
        )


class FakeAgentBanter:
    """Banter fake that returns deterministic direct replies and optional action reactions."""

    def __init__(self, *, reaction: AgentBanterDecision | None = None) -> None:
        """Initialize the fake with an optional human-action reaction."""
        self.reaction = reaction or AgentBanterDecision(
            agent_id=None,
            speech=None,
            reaction={"intent": "no_reaction"},
            confidence=1.0,
        )
        self.direct_requests: list[HumanTableTalkInput] = []
        self.action_events: list[str] = []

    async def respond_to_human_table_talk(
        self,
        request: HumanTableTalkInput,
        public_state: PublicGameState,
    ) -> AgentBanterDecision:
        """Return a deterministic response from the addressed agent."""
        _ = public_state
        self.direct_requests.append(request)
        return AgentBanterDecision(
            agent_id=request.target_agent_id,
            speech="I am listening.",
            reaction={"intent": "table_talk_reply"},
            confidence=0.9,
            emotion="confused",
        )

    async def react_to_human_action(
        self,
        event: GameEvent,
        public_state: PublicGameState,
    ) -> AgentBanterDecision:
        """Return the configured reaction to a human action."""
        _ = public_state
        self.action_events.append(event.event_id)
        return self.reaction


def build_test_runtime(
    public_vision: PublicVisionSource | None = None,
    private_vision: PrivateCardSource | None = None,
    agent_banter_source: AgentBanterSource | None = None,
) -> DashboardRuntime:
    orchestrator = InMemoryOrchestrator()
    runtime_ref: dict[str, DashboardRuntime] = {}

    def snapshot() -> dict[str, Any]:
        return runtime_ref["runtime"].snapshot()

    broadcaster = DashboardEventBroadcaster(snapshot)
    processor = PublicBoardFrameProcessor(
        orchestrator=orchestrator,
        public_vision=public_vision or NoopPublicVision(),
        broadcaster=broadcaster,
    )
    private_processor = PrivateCardsFrameProcessor(
        orchestrator=orchestrator,
        private_cards=private_vision or StaticPrivateCardsVision(),
        broadcaster=broadcaster,
    )
    revealed_processor = RevealedCardsFrameProcessor(
        orchestrator=orchestrator,
        revealed_cards=StaticRevealedCardsVision(),
        broadcaster=broadcaster,
    )
    agent_action_processor = AgentActionProcessor(
        orchestrator=orchestrator,
        decisions=FakeAgentDecisions(),
        broadcaster=broadcaster,
    )
    runtime = DashboardRuntime(
        orchestrator=orchestrator,
        broadcaster=broadcaster,
        board_processor=processor,
        private_cards_processor=private_processor,
        revealed_cards_processor=revealed_processor,
        agent_action_processor=agent_action_processor,
        agent_banter_source=agent_banter_source,
        voice_client_factory=FakeVoiceClient,
    )
    runtime_ref["runtime"] = runtime
    return runtime


def test_index_renders_starter_page():
    client = TestClient(create_app(build_test_runtime()))

    response = client.get("/")

    assert response.status_code == 200
    assert "PokerBot 3000" in response.text
    assert "Table Camera" in response.text
    assert "/static/styles.css" in response.text
    assert "/static/app.js?v=" in response.text


def test_eliza_client_page_renders_thin_client_assets():
    client = TestClient(create_app(build_test_runtime()))

    response = client.get("/clients/eliza")

    assert response.status_code == 200
    assert "Eliza" in response.text
    assert "Noto+Emoji" in response.text
    assert "/static/eliza.js?v=" in response.text


def test_missing_static_file_returns_not_found():
    client = TestClient(create_app(build_test_runtime()))

    response = client.get("/static/missing.css")

    assert response.status_code == 404


def test_removed_voice_transcript_endpoint_identifies_stale_browser_bundle():
    client = TestClient(create_app(build_test_runtime()))

    response = client.post("/api/voice/transcript", json={"text": "call"})

    assert response.status_code == 410
    assert "Stale dashboard JavaScript" in response.json()["detail"]


def test_api_state_returns_public_game_snapshot():
    client = TestClient(create_app(build_test_runtime()))

    response = client.get("/api/state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["hand_id"] == "hand_001"
    assert payload["automation_status"] == "stopped"
    assert payload["waiting_for"] is None
    assert payload["showdown"]["payouts_by_seat"] == {}
    assert payload["players"]["1"]["name"] == "Che"
    assert "reachy" not in payload


def test_api_events_returns_initial_event():
    client = TestClient(create_app(build_test_runtime()))

    response = client.get("/api/events")

    assert response.status_code == 200
    payload = response.json()
    assert payload[0]["event_type"] == "system"


def test_api_start_and_stop_game():
    runtime = build_test_runtime()
    with TestClient(create_app(runtime)) as client:
        start_response = client.post("/api/game/start")

        assert start_response.status_code == 200
        start_payload = start_response.json()
        assert start_payload["accepted"] is True
        assert start_payload["state"]["automation_status"] == "waiting_for_external_input"
        assert start_payload["state"]["waiting_for"]["type"] == "human_action"
        assert start_payload["state"]["pot"] == 30
        assert runtime.board_processor.is_waiting_for_frames is False

        stop_response = client.post("/api/game/stop")

        assert stop_response.status_code == 200
        stop_payload = stop_response.json()
        assert stop_payload["accepted"] is True
        assert stop_payload["state"]["automation_status"] == "stopped"
        assert stop_payload["state"]["waiting_for"] is None
        assert runtime.board_processor.is_waiting_for_frames is False


def test_api_does_not_expose_operator_next_hand_control():
    client = TestClient(create_app(build_test_runtime()))

    response = client.post("/api/game/next-hand")

    assert response.status_code == 404


def test_api_start_queues_orchestrator_speech_for_browser_playback():
    runtime = build_test_runtime()
    with TestClient(create_app(runtime)) as client:
        payload = client.post("/api/game/start").json()

    speech_events = [event for event in payload["events"] if event["event_type"] == "presentation_command"]
    assert speech_events[0]["payload"]["voice"] == "orchestrator"
    assert speech_events[0]["payload"]["speech"].startswith("Move the dealer button to Che.")


def test_runtime_prewarms_orchestrator_voice_for_new_speech_events():
    async def scenario() -> None:
        voice = RecordingVoiceClient()
        runtime = build_test_runtime()
        runtime.voice_client_factory = lambda: voice

        result = await runtime.start_game()
        await asyncio.sleep(0)

        speech_event = next(event for event in result.events if event.event_type == "presentation_command")
        expected = (
            "Move the dealer button to Che. Reachy posts small blind 10. "
            "Eliza posts big blind 20. Deal two cards. Action is on Che."
        )
        assert voice.calls == [expected]
        assert await runtime.synthesize_orchestrator_event(speech_event.event_id) == f"audio:{expected}".encode()
        assert voice.calls == [expected]

    asyncio.run(scenario())


def test_api_human_action_pauses_for_agent_presentation_before_next_action():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    _open_human_action_after_flop(runtime.orchestrator)

    response = client.post(
        "/api/inputs/human-action",
        json={
            "source": "voice",
            "action": {"type": "bet", "amount": 100, "unit": "chips"},
            "raw_transcript": "bet one hundred",
            "confidence": 0.95,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["state"]["pot"] == 260
    assert payload["state"]["waiting_for"]["type"] == "presentation"
    assert payload["state"]["waiting_for"]["agent_id"] == "reachy"
    assert "action_proposed" in {event["event_type"] for event in payload["events"]}
    assert "action_committed" in {event["event_type"] for event in payload["events"]}


def test_api_human_table_talk_responds_without_consuming_action():
    banter = FakeAgentBanter()
    runtime = build_test_runtime(agent_banter_source=banter)
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")

    response = client.post(
        "/api/inputs/human-table-talk",
        json={
            "source": "voice",
            "target_agent_id": "eliza",
            "message": "are you feeling lucky",
            "raw_transcript": "Eliza, are you feeling lucky?",
            "confidence": 0.95,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["state"]["waiting_for"]["type"] == "human_action"
    assert payload["state"]["active_player_seat"] == 1
    assert "human_table_talk" in {event["event_type"] for event in payload["events"]}
    presentation = next(event for event in payload["events"] if event["event_type"] == "presentation_command")
    assert presentation["payload"]["target_client"] == "eliza"
    assert presentation["payload"]["speech"] == "I am listening."
    assert presentation["payload"]["emotion"] == "confused"
    assert presentation["payload"]["blocks_game_flow"] is True
    assert banter.direct_requests[0].target_agent_id == "eliza"


def test_runtime_table_talk_blocks_human_action_until_speech_completes():
    runtime = build_test_runtime(agent_banter_source=FakeAgentBanter())
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    _open_human_action_after_flop(runtime.orchestrator)

    table_talk = client.post(
        "/api/inputs/human-table-talk",
        json={
            "source": "voice",
            "target_agent_id": "eliza",
            "message": "are you nervous",
            "raw_transcript": "Eliza, are you nervous?",
            "confidence": 0.95,
        },
    )
    table_talk_payload = table_talk.json()
    presentation = next(
        event for event in table_talk_payload["events"] if event["event_type"] == "presentation_command"
    )
    assert presentation["payload"]["blocks_game_flow"] is True

    blocked_response = client.post(
        "/api/inputs/human-action",
        json={"source": "voice", "action": {"type": "bet", "amount": 100}},
    )

    assert blocked_response.status_code == 200
    blocked_payload = blocked_response.json()
    assert blocked_payload["accepted"] is False
    assert blocked_payload["reason"] == "Waiting for agent speech to finish before accepting another action."
    assert blocked_payload["state"]["waiting_for"]["type"] == "human_action"
    assert "agent_decision" not in {event["event_type"] for event in blocked_payload["events"]}

    client.post(f"/api/presentation/{presentation['event_id']}/complete")
    response = client.post(
        "/api/inputs/human-action",
        json={"source": "voice", "action": {"type": "bet", "amount": 100}},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert "agent_decision" in {event["event_type"] for event in payload["events"]}
    assert payload["state"]["waiting_for"]["type"] == "presentation"
    assert payload["state"]["waiting_for"]["agent_id"] == "reachy"


def test_runtime_voice_capture_gate_closes_during_blocking_table_talk_response(monkeypatch):
    now = 100.0
    monkeypatch.setattr("pokerbot_3000.app.runtime.time.monotonic", lambda: now)
    runtime = build_test_runtime(agent_banter_source=FakeAgentBanter())
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")

    table_talk = client.post(
        "/api/inputs/human-table-talk",
        json={
            "source": "voice",
            "target_agent_id": "eliza",
            "message": "are you nervous",
            "raw_transcript": "Eliza, are you nervous?",
            "confidence": 0.95,
        },
    )
    presentation = next(
        event for event in table_talk.json()["events"] if event["event_type"] == "presentation_command"
    )

    assert runtime.voice_capture_suppression_reason() == "agent speech is still playing"

    client.post(f"/api/presentation/{presentation['event_id']}/complete")

    assert runtime.voice_capture_suppression_reason() == "agent speech just finished"

    now = 102.0

    assert runtime.voice_capture_suppression_reason() is None


def test_runtime_voice_capture_gate_stays_closed_after_river_agent_handoff():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    _open_human_action_after_flop(runtime.orchestrator)
    runtime.orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "check"}}))
    turn = [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs"), _card("king", "spades")]
    for _ in range(2):
        runtime.orchestrator.record_public_observation(
            PublicTableObservation(board_cards=turn, street_hint=Street.TURN, confidence=0.9)
        )
    _submit_agent_decision_and_complete(runtime.orchestrator, _decision("reachy", "check"))
    _submit_agent_decision_and_complete(runtime.orchestrator, _decision("eliza", "check"))
    runtime.orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "check"}}))
    river = [*turn, _card("9", "spades")]
    for _ in range(2):
        runtime.orchestrator.record_public_observation(
            PublicTableObservation(board_cards=river, street_hint=Street.RIVER, confidence=0.9)
        )

    reachy_events = asyncio.run(runtime.process_pending_agent_actions())
    reachy_presentation = next(
        event
        for event in reachy_events
        if event.event_type == "presentation_command" and event.payload.get("target_client") == ClientId.REACHY
    )
    eliza_events = asyncio.run(runtime.complete_presentation(reachy_presentation.event_id))
    eliza_presentation = next(
        event
        for event in eliza_events
        if event.event_type == "presentation_command" and event.payload.get("target_client") == ClientId.ELIZA
    )

    handoff_events = asyncio.run(runtime.complete_presentation(eliza_presentation.event_id))

    assert any(event.payload.get("intent") == "action_handoff" for event in handoff_events)
    waiting_for = runtime.orchestrator.public_state().waiting_for
    assert waiting_for is not None
    assert waiting_for.type == PendingInputType.HUMAN_ACTION
    assert runtime.voice_capture_suppression_reason() == "agent speech just finished"


def test_runtime_human_action_reaction_does_not_block_agent_decisions():
    reaction = AgentBanterDecision(
        agent_id="reachy",
        speech="Bold move, Che.",
        reaction={"intent": "human_action_reaction"},
        confidence=0.9,
        emotion="confident",
    )
    runtime = build_test_runtime(agent_banter_source=FakeAgentBanter(reaction=reaction))
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    _open_human_action_after_flop(runtime.orchestrator)

    response = client.post(
        "/api/inputs/human-action",
        json={"source": "voice", "action": {"type": "bet", "amount": 100}},
    )

    assert response.status_code == 200
    payload = response.json()
    presentation = next(
        event
        for event in payload["events"]
        if event["event_type"] == "presentation_command"
        and event["payload"].get("intent") == "human_action_reaction"
    )
    assert presentation["payload"]["target_client"] == "reachy"
    assert presentation["payload"]["speech"] == "Bold move, Che."
    assert presentation["payload"]["emotion"] == "confident"
    assert presentation["payload"]["blocks_game_flow"] is False
    assert "agent_decision" in {event["event_type"] for event in payload["events"]}
    assert payload["state"]["waiting_for"]["type"] == "presentation"
    assert payload["state"]["waiting_for"]["agent_id"] == "reachy"


def test_api_thin_client_private_cards_trigger_internal_agent_turn():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    client.post("/api/inputs/human-action", json={"action": {"type": "call"}})

    response = client.post(
        "/api/clients/reachy/private-cards",
        json={
            "agent_id": "reachy",
            "seat": 2,
            "hole_cards": [
                {"rank": "king", "suit": "clubs"},
                {"rank": "king", "suit": "diamonds"},
            ],
            "source": "reachy_camera",
            "confidence": 0.89,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["state"]["pot"] == 60
    assert payload["state"]["waiting_for"]["type"] == "presentation"
    assert payload["state"]["waiting_for"]["agent_id"] == "reachy"
    assert "agent_decision" in {event["event_type"] for event in payload["events"]}


def test_api_thin_client_private_card_frame_triggers_internal_agent_turn():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    client.post("/api/inputs/human-action", json={"action": {"type": "call"}})

    response = client.post(
        "/api/clients/reachy/private-cards/frame",
        json={"source": "reachy_camera", "data_uri": "data:image/png;base64,dGVzdGZyYW1l"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["state"]["waiting_for"]["type"] == "presentation"
    assert payload["state"]["waiting_for"]["agent_id"] == "reachy"
    assert "agent_decision" in {event["event_type"] for event in payload["events"]}


def test_api_private_card_frame_with_no_cards_keeps_reachy_blocked():
    runtime = build_test_runtime(private_vision=StaticPrivateCardsVision(cards=[]))
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    client.post("/api/inputs/human-action", json={"action": {"type": "call"}})

    response = client.post(
        "/api/clients/reachy/private-cards/frame",
        json={"source": "reachy_camera", "data_uri": "data:image/png;base64,dGVzdGZyYW1l"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is False
    assert payload["reason"] == "Expected 2 private cards for Reachy, detected 0."
    assert payload["state"]["waiting_for"]["type"] == "private_cards"
    assert payload["state"]["waiting_for"]["agent_id"] == "reachy"
    assert "agent_decision" not in {event["event_type"] for event in payload["events"]}


def test_api_client_status_update_appears_in_snapshot():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))

    response = client.post(
        "/api/clients/eliza/status",
        json={"connection": "connected", "status": "Eliza browser connected", "detail": "ready"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["client_id"] == "eliza"
    assert payload["connection"] == "connected"
    snapshot = runtime.snapshot()
    eliza = next(status for status in snapshot["client_statuses"] if status["client_id"] == "eliza")
    assert eliza["status"] == "Eliza browser connected"


def test_client_websocket_receives_only_own_private_state():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    runtime.orchestrator.start_game()
    _open_human_action_after_flop(runtime.orchestrator)

    with client.websocket_connect("/ws/clients/eliza") as websocket:
        initial = websocket.receive_json()

    assert initial["client_id"] == "eliza"
    assert [state["agent_id"] for state in initial["private_states"]] == ["eliza"]


def test_websocket_receives_initial_snapshot_and_start_update():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))

    with client.websocket_connect("/ws/events") as websocket:
        initial = websocket.receive_json()
        assert initial["type"] == "snapshot"
        assert initial["state"]["automation_status"] == "stopped"
        assert initial["voice_input"]["state"] == "not_configured"

        client.post("/api/game/start")
        update = websocket.receive_json()

    assert update["type"] == "snapshot"
    assert update["state"]["waiting_for"]["type"] == "human_action"
    assert "game_started" in {event["event_type"] for event in update["events"]}


def test_voice_websocket_queues_browser_pcm_chunks():
    runtime = build_test_runtime()
    runtime.browser_voice_input = BrowserAudioInput()
    client = TestClient(create_app(runtime))

    with client.websocket_connect("/ws/voice/human") as websocket:
        websocket.send_bytes(b"\x00\x01" * 512)
        for _ in range(20):
            if runtime.browser_voice_input.pending_chunk_count:
                break
            time.sleep(0.01)

    assert runtime.browser_voice_input.pending_chunk_count == 1
    assert runtime.browser_voice_input.submitted_chunk_count == 1
    assert runtime.browser_voice_input.submitted_byte_count == 1024
    assert runtime.browser_voice_input.connected is False


def test_voice_websocket_drops_browser_pcm_while_voice_capture_is_suppressed():
    runtime = build_test_runtime(agent_banter_source=FakeAgentBanter())
    runtime.browser_voice_input = BrowserAudioInput()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    client.post(
        "/api/inputs/human-table-talk",
        json={
            "source": "voice",
            "target_agent_id": "eliza",
            "message": "are you nervous",
            "raw_transcript": "Eliza, are you nervous?",
            "confidence": 0.95,
        },
    )

    with client.websocket_connect("/ws/voice/human") as websocket:
        websocket.send_bytes(b"\x00\x01" * 512)
        time.sleep(0.01)

    assert runtime.browser_voice_input.pending_chunk_count == 0
    assert runtime.browser_voice_input.submitted_chunk_count == 0
    assert runtime.browser_voice_input.submitted_byte_count == 0
    assert runtime.browser_voice_input.connected is False


def test_orchestrator_voice_endpoint_returns_audio_for_presentation_event():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    start_payload = client.post("/api/game/start").json()
    event_id = next(
        event["event_id"] for event in start_payload["events"] if event["event_type"] == "presentation_command"
    )

    response = client.get(f"/api/voice/orchestrator/{event_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content.startswith(b"audio:")


def test_eliza_voice_endpoint_returns_agent_audio_for_targeted_presentation_event():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    _open_human_action_after_flop(runtime.orchestrator)
    event = next(
        event
        for event in runtime.orchestrator.events()
        if event.event_type == "presentation_command" and event.payload.get("target_client") == "eliza"
        and isinstance(event.payload.get("speech"), str)
    )

    response = client.get(f"/api/voice/eliza/{event.event_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content.startswith(b"eliza:")


def test_reachy_voice_endpoint_returns_agent_audio_for_targeted_presentation_event():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    _open_human_action_after_flop(runtime.orchestrator)
    event = next(
        event
        for event in runtime.orchestrator.events()
        if event.event_type == "presentation_command" and event.payload.get("target_client") == "reachy"
        and isinstance(event.payload.get("speech"), str)
    )

    response = client.get(f"/api/voice/reachy/{event.event_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content.startswith(b"reachy:")


def test_public_board_frame_submission_advances_recognition_from_browser_image():
    cards = [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")]
    vision = StaticBoardVision(cards)
    runtime = build_test_runtime(vision)
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    client.post("/api/inputs/human-action", json={"action": {"type": "call"}})
    reachy_response = client.post(
        "/api/clients/reachy/private-cards",
        json={
            "agent_id": "reachy",
            "seat": 2,
            "hole_cards": [{"rank": "king", "suit": "clubs"}, {"rank": "king", "suit": "diamonds"}],
            "source": "reachy_camera",
            "confidence": 0.89,
        },
    )
    _complete_first_agent_presentation(client, reachy_response.json()["events"])
    eliza_response = client.post(
        "/api/clients/eliza/private-cards",
        json={
            "agent_id": "eliza",
            "seat": 3,
            "hole_cards": [{"rank": "9", "suit": "clubs"}, {"rank": "9", "suit": "diamonds"}],
            "source": "eliza_browser_webcam",
            "confidence": 0.89,
        },
    )
    _complete_first_agent_presentation(client, eliza_response.json()["events"])

    first_response = client.post(
        "/api/vision/public-board/frame",
        json={"source": "obs_virtual_camera", "data_uri": "data:image/png;base64,dGVzdGZyYW1l"},
    )
    second_response = client.post(
        "/api/vision/public-board/frame",
        json={"source": "obs_virtual_camera", "data_uri": "data:image/png;base64,dGVzdGZyYW1l"},
    )

    assert first_response.status_code == 200
    assert first_response.json()["state"]["board_recognition"]["stable_sample_count"] == 1
    assert second_response.status_code == 200
    assert second_response.json()["state"]["board"] == [
        {"rank": "ace", "suit": "hearts"},
        {"rank": "7", "suit": "diamonds"},
        {"rank": "2", "suit": "clubs"},
    ]
    assert vision.frames[0].source == "obs_virtual_camera"


def test_revealed_cards_frame_submission_advances_current_showdown_reveal():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    _open_human_action_after_flop(runtime.orchestrator)
    runtime.orchestrator.submit_human_action(
        HumanActionInput.model_validate({"source": "voice", "action": {"type": "check"}})
    )
    _commit_turn(runtime.orchestrator)
    runtime.orchestrator.submit_human_action(
        HumanActionInput.model_validate({"source": "voice", "action": {"type": "check"}})
    )
    _commit_river(runtime.orchestrator)
    runtime.orchestrator.submit_human_action(
        HumanActionInput.model_validate({"source": "voice", "action": {"type": "check"}})
    )

    response = client.post(
        "/api/vision/showdown/revealed-cards",
        json={"seat": 2, "source": "seat_2_crop", "data_uri": "data:image/png;base64,dGVzdGZyYW1l"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["state"]["showdown"]["revealed_cards_by_seat"]["2"] == [
        {"rank": "9", "suit": "clubs"},
        {"rank": "9", "suit": "diamonds"},
    ]
    assert payload["state"]["waiting_for"]["seat"] == 3


def _complete_board(orchestrator: InMemoryOrchestrator) -> None:
    flop = [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")]
    turn = [*flop, _card("king", "spades")]
    river = [*turn, _card("9", "clubs")]
    for cards in (flop, flop, turn, turn, river, river):
        orchestrator.record_public_observation(
            PublicTableObservation(
                board_cards=cards,
                street_hint={3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}[len(cards)],
                confidence=0.9,
            )
        )


def _open_human_action_after_flop(orchestrator: InMemoryOrchestrator) -> None:
    flop = [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")]
    orchestrator.submit_human_action(HumanActionInput.model_validate({"action": {"type": "call"}}))
    orchestrator.record_client_private_cards(
        "reachy",
        PrivateCardObservation(
            agent_id="reachy",
            seat=2,
            hole_cards=[_card("king", "clubs"), _card("king", "diamonds")],
            source="reachy_camera",
            confidence=0.89,
        ),
    )
    _submit_agent_decision_and_complete(orchestrator, _decision("reachy", "call"))
    orchestrator.record_client_private_cards(
        "eliza",
        PrivateCardObservation(
            agent_id="eliza",
            seat=3,
            hole_cards=[_card("9", "clubs"), _card("9", "diamonds")],
            source="eliza_browser_webcam",
            confidence=0.89,
        ),
    )
    _submit_agent_decision_and_complete(orchestrator, _decision("eliza", "check"))
    for _ in range(2):
        orchestrator.record_public_observation(
            PublicTableObservation(board_cards=flop, street_hint=Street.FLOP, confidence=0.9)
        )
    _submit_agent_decision_and_complete(orchestrator, _decision("reachy", "check"))
    _submit_agent_decision_and_complete(orchestrator, _decision("eliza", "check"))


def _commit_turn(orchestrator: InMemoryOrchestrator) -> None:
    turn = [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs"), _card("king", "spades")]
    for _ in range(2):
        orchestrator.record_public_observation(
            PublicTableObservation(board_cards=turn, street_hint=Street.TURN, confidence=0.9)
        )
    _submit_agent_decision_and_complete(orchestrator, _decision("reachy", "check"))
    _submit_agent_decision_and_complete(orchestrator, _decision("eliza", "check"))


def _commit_river(orchestrator: InMemoryOrchestrator) -> None:
    river = [
        _card("ace", "hearts"),
        _card("7", "diamonds"),
        _card("2", "clubs"),
        _card("king", "spades"),
        _card("9", "spades"),
    ]
    for _ in range(2):
        orchestrator.record_public_observation(
            PublicTableObservation(board_cards=river, street_hint=Street.RIVER, confidence=0.9)
        )
    _submit_agent_decision_and_complete(orchestrator, _decision("reachy", "check"))
    _submit_agent_decision_and_complete(orchestrator, _decision("eliza", "check"))


def _decision(agent_id: str, action_type: str, amount: int | None = None) -> AgentDecision:
    return AgentDecision(
        agent_id=agent_id,
        action=PokerAction.model_validate({"type": action_type, "amount": amount}),
        speech=f"{agent_id} {action_type}",
        reaction={"intent": "announce_action"},
        confidence=0.9,
    )


def _submit_agent_decision_and_complete(
    orchestrator: InMemoryOrchestrator,
    decision: AgentDecision,
) -> None:
    result = orchestrator.submit_agent_decision(decision)
    presentation = next(
        (
            event
            for event in result.events
            if event.event_type == "presentation_command"
            and event.payload.get("target_client") == decision.agent_id
            and isinstance(event.payload.get("speech"), str)
        ),
        None,
    )
    if presentation is not None:
        orchestrator.complete_presentation(presentation.event_id)


def _complete_first_agent_presentation(client: TestClient, events: list[dict[str, Any]]) -> None:
    presentation = next(
        (
            event
            for event in events
            if event["event_type"] == "presentation_command"
            and event["payload"].get("target_client") in {"reachy", "eliza"}
            and isinstance(event["payload"].get("speech"), str)
        ),
        None,
    )
    if presentation is not None:
        client.post(f"/api/presentation/{presentation['event_id']}/complete")


def _card(rank: str, suit: str) -> Card:
    return Card.model_validate({"rank": rank, "suit": suit})
