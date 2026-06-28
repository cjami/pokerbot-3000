import asyncio
from typing import Any

from fastapi.testclient import TestClient

from pokerbot_3000.app.runtime import DashboardEventBroadcaster, DashboardRuntime, PublicBoardFrameProcessor
from pokerbot_3000.app.server import create_app
from pokerbot_3000.domain.cards import Card
from pokerbot_3000.domain.models import PublicTableObservation, Street
from pokerbot_3000.orchestrator import InMemoryOrchestrator
from pokerbot_3000.ports.llm import ImageFrame
from pokerbot_3000.ports.perception import PublicVisionSource


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


class FakeVoiceClient:
    """Voice fake that returns deterministic bytes."""

    async def synthesize_orchestrator(self, text: str) -> bytes:
        """Return fake MPEG bytes for a speech line."""
        return f"audio:{text}".encode()


class RecordingVoiceClient:
    """Voice fake that records synthesis requests."""

    def __init__(self) -> None:
        """Initialize the request log."""
        self.calls: list[str] = []

    async def synthesize_orchestrator(self, text: str) -> bytes:
        """Record the requested speech and return fake MPEG bytes."""
        self.calls.append(text)
        return f"audio:{text}".encode()


def build_test_runtime(public_vision: PublicVisionSource | None = None) -> DashboardRuntime:
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
    runtime = DashboardRuntime(
        orchestrator=orchestrator,
        broadcaster=broadcaster,
        board_processor=processor,
        voice_client_factory=FakeVoiceClient,
    )
    runtime_ref["runtime"] = runtime
    return runtime


def test_index_renders_starter_page():
    client = TestClient(create_app(build_test_runtime()))

    response = client.get("/")

    assert response.status_code == 200
    assert "Pokerbot 3000" in response.text
    assert "Table State" in response.text
    assert "/static/styles.css" in response.text
    assert "/static/app.js" in response.text


def test_missing_static_file_returns_not_found():
    client = TestClient(create_app(build_test_runtime()))

    response = client.get("/static/missing.css")

    assert response.status_code == 404


def test_api_state_returns_public_game_snapshot():
    client = TestClient(create_app(build_test_runtime()))

    response = client.get("/api/state")

    assert response.status_code == 200
    payload = response.json()
    assert payload["hand_id"] == "hand_001"
    assert payload["automation_status"] == "stopped"
    assert payload["waiting_for"] is None
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
        assert start_payload["state"]["waiting_for"]["type"] == "public_board_cards"
        assert runtime.board_processor.is_waiting_for_frames is True

        stop_response = client.post("/api/game/stop")

        assert stop_response.status_code == 200
        stop_payload = stop_response.json()
        assert stop_payload["accepted"] is True
        assert stop_payload["state"]["automation_status"] == "stopped"
        assert stop_payload["state"]["waiting_for"] is None
        assert runtime.board_processor.is_waiting_for_frames is False


def test_api_start_queues_orchestrator_speech_for_browser_playback():
    runtime = build_test_runtime()
    with TestClient(create_app(runtime)) as client:
        payload = client.post("/api/game/start").json()

    speech_events = [event for event in payload["events"] if event["event_type"] == "presentation_command"]
    assert speech_events[0]["payload"]["voice"] == "orchestrator"
    assert speech_events[0]["payload"]["speech"] == "Please lay out the flop."


def test_runtime_prewarms_orchestrator_voice_for_new_speech_events():
    async def scenario() -> None:
        voice = RecordingVoiceClient()
        runtime = build_test_runtime()
        runtime.voice_client_factory = lambda: voice

        result = await runtime.start_game()
        await asyncio.sleep(0)

        speech_event = next(event for event in result.events if event.event_type == "presentation_command")
        assert voice.calls == ["Please lay out the flop."]
        assert await runtime.synthesize_orchestrator_event(speech_event.event_id) == b"audio:Please lay out the flop."
        assert voice.calls == ["Please lay out the flop."]

    asyncio.run(scenario())


def test_api_human_action_advances_until_eliza_input_needed():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    _complete_board(runtime.orchestrator)

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
    assert payload["state"]["pot"] == 100
    assert payload["state"]["waiting_for"]["type"] == "private_cards"
    assert payload["state"]["waiting_for"]["agent_id"] == "eliza"


def test_api_thin_client_private_cards_trigger_internal_agent_turn():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")
    _complete_board(runtime.orchestrator)
    client.post(
        "/api/inputs/human-action",
        json={"source": "voice", "action": {"type": "bet", "amount": 100}},
    )

    response = client.post(
        "/api/clients/eliza/private-cards",
        json={
            "agent_id": "eliza",
            "seat": 3,
            "hole_cards": [
                {"rank": "9", "suit": "clubs"},
                {"rank": "9", "suit": "diamonds"},
            ],
            "source": "eliza_browser_webcam",
            "confidence": 0.89,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["accepted"] is True
    assert payload["state"]["pot"] == 200
    assert payload["state"]["waiting_for"]["agent_id"] == "reachy"
    assert "agent_decision" in {event["event_type"] for event in payload["events"]}


def test_websocket_receives_initial_snapshot_and_start_update():
    runtime = build_test_runtime()
    client = TestClient(create_app(runtime))

    with client.websocket_connect("/ws/events") as websocket:
        initial = websocket.receive_json()
        assert initial["type"] == "snapshot"
        assert initial["state"]["automation_status"] == "stopped"

        client.post("/api/game/start")
        update = websocket.receive_json()

    assert update["type"] == "snapshot"
    assert update["state"]["waiting_for"]["type"] == "public_board_cards"
    assert "game_started" in {event["event_type"] for event in update["events"]}


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


def test_public_board_frame_submission_advances_recognition_from_browser_image():
    cards = [_card("ace", "hearts"), _card("7", "diamonds"), _card("2", "clubs")]
    vision = StaticBoardVision(cards)
    runtime = build_test_runtime(vision)
    client = TestClient(create_app(runtime))
    client.post("/api/game/start")

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


def _card(rank: str, suit: str) -> Card:
    return Card.model_validate({"rank": rank, "suit": suit})
