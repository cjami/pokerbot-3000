from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from pokerbot_3000.reachy_bridge import (
    BridgeConfig,
    ReachyBridge,
    ReachyDaemonHttpAdapter,
    ReachyMiniAdapter,
    ReachySdkConfig,
)


class _FakeHttpClient:
    def __init__(self, actions: list[str] | None = None) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.state: dict[str, Any] = {"waiting_for": None}
        self.events: list[dict[str, Any]] = []
        self.actions = actions

    def get_json(self, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        if self.actions is not None:
            self.actions.append(f"get:{path}")
        if path == "/api/state":
            return self.state
        if path == "/api/events?limit=25":
            return self.events
        raise AssertionError(path)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((path, payload))
        if self.actions is not None:
            self.actions.append(f"post:{path}:{payload.get('connection', '')}")
        return {"accepted": True, "reason": "ok"}


class _FakeReachy:
    def __init__(self, actions: list[str] | None = None) -> None:
        self.presentations: list[tuple[str, str | None]] = []
        self.wake_count = 0
        self.capture_count = 0
        self.actions = actions

    def wake_up(self) -> None:
        self.wake_count += 1
        if self.actions is not None:
            self.actions.append("wake_up")

    def capture_private_cards(self) -> str:
        self.capture_count += 1
        return "data:image/png;base64,dGVzdA=="

    def perform(self, emotion: str, speech: str | None) -> None:
        self.presentations.append((emotion, speech))
        if self.actions is not None:
            self.actions.append(f"perform:{emotion}:{speech or ''}")


def test_reachy_bridge_wakes_after_connect_before_polling(monkeypatch):
    actions: list[str] = []
    http = _FakeHttpClient(actions)
    reachy = _FakeReachy(actions)
    bridge = ReachyBridge(config=BridgeConfig(poll_seconds=0.01), http=http, reachy=reachy)

    def stop_after_first_poll(seconds: float) -> None:
        actions.append(f"sleep:{seconds}")
        raise KeyboardInterrupt

    monkeypatch.setattr("pokerbot_3000.reachy_bridge.time.sleep", stop_after_first_poll)

    with pytest.raises(KeyboardInterrupt):
        bridge.start()

    assert reachy.wake_count == 1
    assert reachy.presentations == []
    assert actions == [
        "post:/api/clients/reachy/status:connected",
        "wake_up",
        "get:/api/state",
        "get:/api/events?limit=25",
        "sleep:0.01",
        "post:/api/clients/reachy/status:disconnected",
    ]


def test_reachy_bridge_can_skip_wake_on_connect(monkeypatch):
    actions: list[str] = []
    http = _FakeHttpClient(actions)
    reachy = _FakeReachy(actions)
    bridge = ReachyBridge(
        config=BridgeConfig(poll_seconds=0.01, wake_on_connect=False),
        http=http,
        reachy=reachy,
    )

    def stop_after_first_poll(seconds: float) -> None:
        actions.append(f"sleep:{seconds}")
        raise KeyboardInterrupt

    monkeypatch.setattr("pokerbot_3000.reachy_bridge.time.sleep", stop_after_first_poll)

    with pytest.raises(KeyboardInterrupt):
        bridge.start()

    assert reachy.presentations == []
    assert reachy.wake_count == 0
    assert "wake_up" not in actions
    assert actions == [
        "post:/api/clients/reachy/status:connected",
        "get:/api/state",
        "get:/api/events?limit=25",
        "sleep:0.01",
        "post:/api/clients/reachy/status:disconnected",
    ]


def test_reachy_bridge_submits_private_card_frame_when_requested():
    http = _FakeHttpClient()
    http.state = {"waiting_for": {"type": "private_cards", "agent_id": "reachy"}}
    reachy = _FakeReachy()
    bridge = ReachyBridge(
        config=BridgeConfig(),
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


def test_reachy_daemon_http_adapter_posts_symbolic_movement(monkeypatch):
    captured: dict[str, object] = {}

    class FakeDaemonHttp:
        def __init__(self, base_url: str) -> None:
            captured["base_url"] = base_url

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            captured["post"] = (path, payload)
            return {"uuid": "move-1"}

    monkeypatch.setattr("pokerbot_3000.reachy_bridge.UrllibBridgeHttpClient", FakeDaemonHttp)

    adapter = ReachyDaemonHttpAdapter("http://reachy-mini.local:8000/")

    adapter.perform("confident", None)

    assert captured["base_url"] == "http://reachy-mini.local:8000/"
    path, payload = cast("tuple[str, dict[str, Any]]", captured["post"])
    assert path == "/api/move/goto"
    assert payload == {
        "head_pose": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.012,
            "roll": pytest.approx(-0.13962634015954636),
            "pitch": 0.0,
            "yaw": 0.0,
        },
        "antennas": [pytest.approx(0.7853981633974483), pytest.approx(0.7853981633974483)],
        "body_yaw": pytest.approx(0.20943951023931953),
        "duration": 0.8,
        "interpolation": "cartoon",
    }


def test_reachy_daemon_http_adapter_uses_builtin_wake_up(monkeypatch):
    captured: dict[str, object] = {}

    class FakeDaemonHttp:
        def __init__(self, base_url: str) -> None:
            captured["base_url"] = base_url

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            captured["post"] = (path, payload)
            return {"uuid": "wake-1"}

    monkeypatch.setattr("pokerbot_3000.reachy_bridge.UrllibBridgeHttpClient", FakeDaemonHttp)

    adapter = ReachyDaemonHttpAdapter("http://reachy-mini.local:8000/")

    adapter.wake_up()

    assert captured == {
        "base_url": "http://reachy-mini.local:8000/",
        "post": ("/api/move/play/wake_up", {}),
    }


def test_reachy_mini_adapter_passes_sdk_connection_config(monkeypatch):
    captured: dict[str, object] = {}

    class FakeMini:
        def wake_up(self) -> None:
            captured["woke"] = True

    class FakeContext:
        def __enter__(self) -> FakeMini:
            return FakeMini()

        def __exit__(self, *_args: object) -> None:
            captured["closed"] = True

    def fake_reachy_mini(**kwargs: object) -> FakeContext:
        captured["kwargs"] = kwargs
        return FakeContext()

    monkeypatch.setitem(sys.modules, "reachy_mini", SimpleNamespace(ReachyMini=fake_reachy_mini))

    adapter = ReachyMiniAdapter(
        ReachySdkConfig(connection_mode="network", media_backend="no_media", timeout_seconds=30.0)
    )
    adapter.wake_up()
    adapter.close()

    assert captured == {
        "kwargs": {"connection_mode": "network", "media_backend": "no_media", "timeout": 30.0},
        "woke": True,
        "closed": True,
    }
