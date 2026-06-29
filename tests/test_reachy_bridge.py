from __future__ import annotations

from typing import Any

from pokerbot_3000.reachy_bridge import BridgeConfig, ReachyBridge


class _FakeHttpClient:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.state: dict[str, Any] = {"waiting_for": None}
        self.events: list[dict[str, Any]] = []

    def get_json(self, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        if path == "/api/state":
            return self.state
        if path == "/api/events?limit=25":
            return self.events
        raise AssertionError(path)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((path, payload))
        return {"accepted": True, "reason": "ok"}


class _FakeReachy:
    def __init__(self) -> None:
        self.presentations: list[tuple[str, str | None]] = []
        self.capture_count = 0

    def capture_private_cards(self) -> str:
        self.capture_count += 1
        return "data:image/png;base64,dGVzdA=="

    def perform(self, emotion: str, speech: str | None) -> None:
        self.presentations.append((emotion, speech))


def test_reachy_bridge_submits_private_card_frame_when_requested():
    http = _FakeHttpClient()
    http.state = {"waiting_for": {"type": "private_cards", "agent_id": "reachy"}}
    reachy = _FakeReachy()
    bridge = ReachyBridge(
        config=BridgeConfig(manual_confirm=False),
        http=http,
        reachy=reachy,
    )

    bridge.tick()

    assert reachy.capture_count == 1
    assert http.posts == [
        (
            "/api/clients/reachy/private-cards/frame",
            {"source": "reachy_private_camera", "data_uri": "data:image/png;base64,dGVzdA=="},
        )
    ]


def test_reachy_bridge_performs_targeted_presentation_once():
    http = _FakeHttpClient()
    http.events = [
        {
            "event_id": "evt_1",
            "payload": {"target_client": "reachy", "emotion": "confident", "speech": "Reachy calls."},
        }
    ]
    reachy = _FakeReachy()
    bridge = ReachyBridge(config=BridgeConfig(manual_confirm=False), http=http, reachy=reachy)

    bridge.tick()
    bridge.tick()

    assert reachy.presentations == [("confident", "Reachy calls.")]
