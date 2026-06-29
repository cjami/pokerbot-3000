from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

from pokerbot_3000.reachy_bridge import (
    BridgeConfig,
    BridgeError,
    ReachyBridge,
    ReachyDaemonHttpAdapter,
    ReachyMiniAdapter,
    ReachySdkConfig,
    _build_reachy_adapter,
)


class _FakeHttpClient:
    def __init__(
        self,
        actions: list[str] | None = None,
        private_frame_results: list[dict[str, Any]] | None = None,
    ) -> None:
        self.posts: list[tuple[str, dict[str, Any]]] = []
        self.state: dict[str, Any] = {"waiting_for": None}
        self.events: list[dict[str, Any]] = []
        self.audio_by_path: dict[str, bytes] = {}
        self.actions = actions
        self.private_frame_results = private_frame_results or []

    def get_json(self, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        if self.actions is not None:
            self.actions.append(f"get:{path}")
        if path == "/api/state":
            return self.state
        if path == "/api/events?limit=25":
            return self.events
        raise AssertionError(path)

    def get_bytes(self, path: str) -> bytes:
        if self.actions is not None:
            self.actions.append(f"get-bytes:{path}")
        return self.audio_by_path.get(path, b"reachy-audio")

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.posts.append((path, payload))
        if self.actions is not None:
            self.actions.append(f"post:{path}:{payload.get('connection', '')}")
        if path == "/api/clients/reachy/private-cards/frame" and self.private_frame_results:
            return self.private_frame_results.pop(0)
        return {"accepted": True, "reason": "ok"}


class _FakeReachy:
    def __init__(self, actions: list[str] | None = None, capture_error: str | None = None) -> None:
        self.presentations: list[tuple[str, str | None, str | None]] = []
        self.voice_audio: list[bytes | None] = []
        self.wake_count = 0
        self.capture_count = 0
        self.actions = actions
        self.capture_error = capture_error

    def wake_up(self) -> None:
        self.wake_count += 1
        if self.actions is not None:
            self.actions.append("wake_up")

    def capture_private_cards(self) -> str:
        self.capture_count += 1
        if self.capture_error is not None:
            raise BridgeError(self.capture_error)
        return "data:image/png;base64,dGVzdA=="

    def perform(
        self,
        emotion: str,
        speech: str | None,
        voice_audio: bytes | None = None,
        gesture: str | None = None,
    ) -> None:
        self.presentations.append((emotion, speech, gesture))
        self.voice_audio.append(voice_audio)
        if self.actions is not None:
            self.actions.append(f"perform:{emotion}:{speech or ''}:{gesture or ''}")


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
        "get:/api/events?limit=25",
        "sleep:0.01",
        "post:/api/clients/reachy/status:disconnected",
    ]


def test_reachy_bridge_does_not_capture_private_cards_from_passive_state():
    http = _FakeHttpClient()
    http.state = {"waiting_for": {"type": "private_cards", "agent_id": "reachy"}}
    reachy = _FakeReachy()
    bridge = ReachyBridge(config=BridgeConfig(), http=http, reachy=reachy)

    bridge.tick()

    assert reachy.capture_count == 0
    assert http.posts == []


def test_reachy_bridge_submits_private_card_frame_for_targeted_request():
    http = _FakeHttpClient()
    http.events = [
        {
            "event_id": "evt_cards",
            "payload": {
                "target_client": "reachy",
                "intent": "request_private_cards",
                "emotion": "calm",
                "gesture": "look_down",
            },
        }
    ]
    reachy = _FakeReachy()
    bridge = ReachyBridge(config=BridgeConfig(capture_settle_seconds=0), http=http, reachy=reachy)

    bridge.tick()
    bridge.tick()

    assert reachy.capture_count == 1
    assert reachy.presentations == [("calm", None, "look_down")]
    assert http.posts == [
        (
            "/api/clients/reachy/private-cards/frame",
            {"source": "reachy_private_camera", "data_uri": "data:image/png;base64,dGVzdA=="},
        )
    ]


def test_reachy_bridge_retries_private_card_frames_until_accepted():
    http = _FakeHttpClient(
        private_frame_results=[
            {"accepted": False, "reason": "Expected 2 private cards for Reachy, detected 0."},
            {"accepted": True, "reason": "ok"},
        ]
    )
    http.events = [
        {
            "event_id": "evt_cards",
            "payload": {
                "target_client": "reachy",
                "intent": "request_private_cards",
                "emotion": "calm",
                "gesture": "look_down",
            },
        }
    ]
    reachy = _FakeReachy()
    bridge = ReachyBridge(config=BridgeConfig(capture_settle_seconds=0), http=http, reachy=reachy)

    bridge.tick()
    bridge.tick()
    bridge.tick()

    assert reachy.capture_count == 2
    assert reachy.presentations == [("calm", None, "look_down")]
    assert [post[0] for post in http.posts] == [
        "/api/clients/reachy/private-cards/frame",
        "/api/clients/reachy/status",
        "/api/clients/reachy/private-cards/frame",
    ]
    assert http.posts[1][1] == {
        "connection": "connected",
        "status": "Reachy private-card frame pending",
        "detail": "Expected 2 private cards for Reachy, detected 0.",
    }


def test_reachy_bridge_reports_private_card_capture_error_without_stopping():
    http = _FakeHttpClient()
    http.events = [
        {
            "event_id": "evt_cards",
            "payload": {"target_client": "reachy", "intent": "request_private_cards", "emotion": "calm"},
        }
    ]
    reachy = _FakeReachy(capture_error="No camera in daemon mode.")
    bridge = ReachyBridge(config=BridgeConfig(capture_settle_seconds=0), http=http, reachy=reachy)

    bridge.tick()
    bridge.tick()

    assert reachy.capture_count == 2
    assert http.posts == [
        (
            "/api/clients/reachy/status",
            {
                "connection": "error",
                "status": "Reachy private-card capture unavailable",
                "detail": "No camera in daemon mode.",
            },
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
    http.audio_by_path["/api/voice/reachy/evt_1"] = b"mp3-bytes"
    reachy = _FakeReachy()
    bridge = ReachyBridge(config=BridgeConfig(), http=http, reachy=reachy)

    bridge.tick()
    bridge.tick()

    assert reachy.presentations == [("confident", "Reachy calls.", None)]
    assert reachy.voice_audio == [b"mp3-bytes"]


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


def test_reachy_daemon_http_adapter_enables_motors_and_posts_ready_pose(monkeypatch):
    posts: list[tuple[str, dict[str, Any]]] = []
    captured: dict[str, object] = {}

    class FakeDaemonHttp:
        def __init__(self, base_url: str) -> None:
            captured["base_url"] = base_url

        def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            posts.append((path, payload))
            return {"uuid": "wake-1"}

    monkeypatch.setattr("pokerbot_3000.reachy_bridge.UrllibBridgeHttpClient", FakeDaemonHttp)

    adapter = ReachyDaemonHttpAdapter("http://reachy-mini.local:8000/")

    adapter.wake_up()

    assert captured["base_url"] == "http://reachy-mini.local:8000/"
    assert posts == [
        ("/api/motors/set_mode/enabled", {}),
        (
            "/api/move/goto",
            {
                "head_pose": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
                "antennas": [0.0, 0.0],
                "body_yaw": 0.0,
                "duration": 2.0,
                "interpolation": "cartoon",
            },
        ),
    ]


def test_reachy_mini_adapter_passes_sdk_connection_config(monkeypatch):
    captured: dict[str, object] = {}

    class FakeMini:
        def enable_motors(self) -> None:
            captured["motors_enabled"] = True

        def goto_target(self, **kwargs: object) -> None:
            captured["goto"] = kwargs

    class FakeContext:
        def __enter__(self) -> FakeMini:
            return FakeMini()

        def __exit__(self, *_args: object) -> None:
            captured["closed"] = True

    def fake_reachy_mini(**kwargs: object) -> FakeContext:
        captured["kwargs"] = kwargs
        return FakeContext()

    monkeypatch.setitem(sys.modules, "reachy_mini", SimpleNamespace(ReachyMini=fake_reachy_mini))
    monkeypatch.setitem(
        sys.modules,
        "reachy_mini.reachy_mini",
        SimpleNamespace(InterpolationTechnique=SimpleNamespace(CARTOON="cartoon")),
    )
    monkeypatch.setitem(
        sys.modules,
        "reachy_mini.utils",
        SimpleNamespace(create_head_pose=lambda **_kwargs: "ready-pose"),
    )

    adapter = ReachyMiniAdapter(
        ReachySdkConfig(
            connection_mode="network",
            host="10.0.0.39",
            port=8000,
            media_backend="no_media",
            timeout_seconds=30.0,
        )
    )
    adapter.wake_up()
    adapter.close()

    goto = cast("dict[str, object]", captured["goto"])
    assert captured == {
        "kwargs": {
            "connection_mode": "network",
            "host": "10.0.0.39",
            "port": 8000,
            "media_backend": "no_media",
            "timeout": 30.0,
        },
        "motors_enabled": True,
        "goto": goto,
        "closed": True,
    }
    assert goto["head"] == "ready-pose"
    assert goto["duration"] == 2.0


def test_reachy_daemon_url_configures_sdk_network_address(monkeypatch):
    captured: dict[str, object] = {}

    class FakeReachyMiniAdapter:
        def __init__(self, config: ReachySdkConfig) -> None:
            captured["config"] = config

    monkeypatch.setattr("pokerbot_3000.reachy_bridge.ReachyMiniAdapter", FakeReachyMiniAdapter)

    adapter = _build_reachy_adapter(
        argparse.Namespace(
            console_only=False,
            reachy_daemon_url="http://10.0.0.39:8000/",
            reachy_connection_mode="auto",
            reachy_media_backend="default",
            reachy_timeout=15.0,
        )
    )

    config = cast("ReachySdkConfig", captured["config"])
    assert isinstance(adapter, FakeReachyMiniAdapter)
    assert config == ReachySdkConfig(connection_mode="network", host="10.0.0.39", port=8000)
