"""Thin Reachy Mini bridge for Pokerbot 3000."""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Literal, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from pokerbot_3000.domain.models import ClientConnectionState

REACHY_CLIENT_ID = "reachy"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/"
DEFAULT_REACHY_DAEMON_URL = "http://reachy-mini.local:8000/"
DEFAULT_POLL_SECONDS = 1.0
DEFAULT_SOURCE = "reachy_private_camera"
DEFAULT_REACHY_VOICE = "Aiden"
DEFAULT_REACHY_CONNECTION_MODE = "auto"
DEFAULT_REACHY_MEDIA_BACKEND = "default"
DEFAULT_REACHY_TIMEOUT_SECONDS = 15.0

ReachyConnectionMode = Literal["auto", "localhost_only", "network"]


class BridgeError(RuntimeError):
    """Raised when the bridge cannot communicate with the app or robot."""


class BridgeHttpClient(Protocol):
    """Minimal HTTP client used by the bridge."""

    def get_json(self, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        """Return a JSON response from a GET request."""

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a JSON response from a POST request."""


class ReachyAdapter(Protocol):
    """Robot operations needed by the poker bridge."""

    def wake_up(self) -> None:
        """Move Reachy out of its sleeping posture."""

    def capture_private_cards(self) -> str:
        """Return a JPEG or PNG data URI of Reachy's private cards."""

    def perform(self, emotion: str, speech: str | None) -> None:
        """Perform one symbolic emotion and speech line."""


@dataclass(frozen=True, slots=True)
class BridgeConfig:
    """Runtime settings for the Reachy bridge."""

    base_url: str = DEFAULT_BASE_URL
    poll_seconds: float = DEFAULT_POLL_SECONDS
    source: str = DEFAULT_SOURCE
    manual_confirm: bool = False
    wake_on_connect: bool = True


class UrllibBridgeHttpClient:
    """Small stdlib JSON HTTP client."""

    def __init__(self, base_url: str) -> None:
        """Create a client rooted at the FastAPI app URL."""
        self._base_url = base_url if base_url.endswith("/") else f"{base_url}/"

    def get_json(self, path: str) -> dict[str, Any] | list[dict[str, Any]]:
        """Return JSON from a GET request."""
        response = self._request_json("GET", path)
        if not isinstance(response, dict | list):
            msg = "Expected JSON object or array response."
            raise BridgeError(msg)
        return cast("dict[str, Any] | list[dict[str, Any]]", response)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Return JSON from a POST request."""
        response = self._request_json("POST", path, payload)
        if not isinstance(response, dict):
            msg = "Expected JSON object response."
            raise BridgeError(msg)
        return cast("dict[str, Any]", response)

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> object:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(  # noqa: S310
            url=urljoin(self._base_url, path.lstrip("/")),
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "pokerbot-reachy-bridge/0.1"},
            method=method,
        )
        try:
            with urlopen(request, timeout=20.0) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            msg = f"Pokerbot API returned HTTP {exc.code}: {detail}"
            raise BridgeError(msg) from exc
        except (URLError, json.JSONDecodeError) as exc:
            msg = f"Could not reach Pokerbot API: {exc}"
            raise BridgeError(msg) from exc


class ConsoleReachyAdapter:
    """Fallback adapter for bridge dry-runs without Reachy hardware."""

    def wake_up(self) -> None:
        """Print startup wake commands when no robot adapter is available."""
        print("Reachy wake: console-only mode")

    def capture_private_cards(self) -> str:
        """Return no frame in console-only mode."""
        msg = "Reachy SDK is not active; cannot capture private cards."
        raise BridgeError(msg)

    def perform(self, emotion: str, speech: str | None) -> None:
        """Print presentation commands when no robot adapter is available."""
        line = speech or ""
        print(f"Reachy presentation: emotion={emotion} voice={DEFAULT_REACHY_VOICE} speech={line}")


@dataclass(frozen=True, slots=True)
class ReachySdkConfig:
    """Connection settings for the Reachy Mini Python SDK."""

    connection_mode: ReachyConnectionMode = DEFAULT_REACHY_CONNECTION_MODE
    media_backend: str = DEFAULT_REACHY_MEDIA_BACKEND
    timeout_seconds: float = DEFAULT_REACHY_TIMEOUT_SECONDS


class ReachyMiniAdapter:
    """Reachy Mini SDK adapter."""

    def __init__(self, config: ReachySdkConfig | None = None) -> None:
        """Connect to Reachy Mini using the Python SDK."""
        config = config or ReachySdkConfig()
        try:
            reachy_module = importlib.import_module("reachy_mini")
        except ImportError as exc:
            msg = "Install the Reachy extras with `uv sync --extra reachy` to use the robot bridge."
            raise BridgeError(msg) from exc

        self._context = reachy_module.ReachyMini(
            connection_mode=config.connection_mode,
            media_backend=config.media_backend,
            timeout=config.timeout_seconds,
        )
        self._mini = self._context.__enter__()

    def close(self) -> None:
        """Release the SDK context."""
        self._context.__exit__(None, None, None)

    def wake_up(self) -> None:
        """Run the SDK's built-in wake-up behavior."""
        self._mini.wake_up()

    def capture_private_cards(self) -> str:
        """Capture the current Reachy camera frame as a PNG data URI."""
        try:
            image_module = importlib.import_module("PIL.Image")
        except ImportError as exc:
            msg = "Install the Reachy extras with `uv sync --extra reachy` to encode camera frames."
            raise BridgeError(msg) from exc

        frame = self._mini.media.get_frame()
        if frame is None:
            msg = "Reachy camera did not return a frame."
            raise BridgeError(msg)
        buffer = BytesIO()
        image_module.fromarray(frame).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def perform(self, emotion: str, speech: str | None) -> None:
        """Perform one symbolic movement and rely on Reachy's default voice path externally."""
        self._move(emotion)
        if speech:
            print(f"Reachy voice ({DEFAULT_REACHY_VOICE}): {speech}")

    def _move(self, emotion: str) -> None:
        try:
            np = importlib.import_module("numpy")
            reachy_core = importlib.import_module("reachy_mini.reachy_mini")
            reachy_utils = importlib.import_module("reachy_mini.utils")
        except ImportError as exc:
            msg = "Reachy movement dependencies are unavailable."
            raise BridgeError(msg) from exc

        pose = _movement_for_emotion(emotion)
        self._mini.goto_target(
            head=reachy_utils.create_head_pose(z=pose.z_mm, roll=pose.roll_deg, degrees=True, mm=True),
            antennas=np.deg2rad([pose.left_antenna_deg, pose.right_antenna_deg]),
            body_yaw=np.deg2rad(pose.body_yaw_deg),
            duration=pose.duration,
            method=reachy_core.InterpolationTechnique.CARTOON,
        )


class ReachyDaemonHttpAdapter:
    """Reachy Mini wireless daemon REST API adapter."""

    def __init__(self, daemon_url: str) -> None:
        """Create an adapter rooted at the Reachy Mini daemon URL."""
        self._http = UrllibBridgeHttpClient(daemon_url)

    def capture_private_cards(self) -> str:
        """Private-card capture is not exposed by the daemon REST API."""
        msg = (
            "Reachy daemon HTTP mode cannot capture camera frames. "
            "Use SDK mode with a supported media backend for private-card capture."
        )
        raise BridgeError(msg)

    def wake_up(self) -> None:
        """Run the daemon's built-in wake-up behavior."""
        self._http.post_json("/api/move/play/wake_up", {})

    def perform(self, emotion: str, speech: str | None) -> None:
        """Perform one symbolic movement through the daemon REST API."""
        pose = _movement_for_emotion(emotion)
        self._http.post_json("/api/move/goto", _daemon_goto_payload(pose))
        if speech:
            print(f"Reachy voice ({DEFAULT_REACHY_VOICE}): {speech}")


@dataclass(frozen=True, slots=True)
class MovementPose:
    """Symbolic Reachy movement recipe."""

    z_mm: float
    roll_deg: float
    left_antenna_deg: float
    right_antenna_deg: float
    body_yaw_deg: float
    duration: float = 0.8


def _movement_for_emotion(emotion: str) -> MovementPose:
    poses = {
        "calm": MovementPose(5, 0, 20, 20, 0),
        "confident": MovementPose(12, -8, 45, 45, 12),
        "celebrate": MovementPose(15, 10, 70, 70, -15, duration=1.0),
        "confused": MovementPose(4, 18, 15, 55, 0),
        "sad": MovementPose(-8, 0, -20, -20, 0),
    }
    return poses.get(emotion, poses["confused"])


def _daemon_goto_payload(pose: MovementPose) -> dict[str, object]:
    return {
        "head_pose": {
            "x": 0.0,
            "y": 0.0,
            "z": pose.z_mm / 1000,
            "roll": math.radians(pose.roll_deg),
            "pitch": 0.0,
            "yaw": 0.0,
        },
        "antennas": [math.radians(pose.left_antenna_deg), math.radians(pose.right_antenna_deg)],
        "body_yaw": math.radians(pose.body_yaw_deg),
        "duration": pose.duration,
        "interpolation": "cartoon",
    }


@dataclass(slots=True)
class ReachyBridge:
    """Poll the orchestrator and perform Reachy thin-client duties."""

    config: BridgeConfig
    http: BridgeHttpClient
    reachy: ReachyAdapter
    seen_event_ids: set[str] = field(default_factory=set)

    def start(self) -> None:
        """Run the bridge until interrupted."""
        self._post_status(ClientConnectionState.CONNECTED, "Reachy bridge connected")
        try:
            if self.config.wake_on_connect:
                self.reachy.wake_up()
            while True:
                self.tick()
                time.sleep(self.config.poll_seconds)
        finally:
            self._post_status(ClientConnectionState.DISCONNECTED, "Reachy bridge stopped")

    def tick(self) -> None:
        """Run one polling iteration."""
        state = self._state()
        self._maybe_capture_private_cards(state)
        for event in self._events():
            self._maybe_perform_event(event)

    def _state(self) -> dict[str, Any]:
        state = self.http.get_json("/api/state")
        if not isinstance(state, dict):
            msg = "Expected state response to be a JSON object."
            raise BridgeError(msg)
        return state

    def _events(self) -> list[dict[str, Any]]:
        events = self.http.get_json("/api/events?limit=25")
        if not isinstance(events, list):
            msg = "Expected events response to be a JSON array."
            raise BridgeError(msg)
        return events

    def _maybe_capture_private_cards(self, state: dict[str, Any]) -> None:
        waiting_for = state.get("waiting_for")
        if not isinstance(waiting_for, dict):
            return
        if waiting_for.get("type") != "private_cards" or waiting_for.get("agent_id") != REACHY_CLIENT_ID:
            return
        if self.config.manual_confirm:
            input("Show Reachy's cards, then press Enter to capture.")
        data_uri = self.reachy.capture_private_cards()
        result = self.http.post_json(
            f"/api/clients/{REACHY_CLIENT_ID}/private-cards/frame",
            {"source": self.config.source, "data_uri": data_uri},
        )
        if not result.get("accepted"):
            self._post_status(
                ClientConnectionState.ERROR,
                "Reachy private-card frame rejected",
                str(result.get("reason")),
            )

    def _maybe_perform_event(self, event: dict[str, Any]) -> None:
        event_id = event.get("event_id")
        payload = event.get("payload")
        if not isinstance(event_id, str) or not isinstance(payload, dict):
            return
        if event_id in self.seen_event_ids or payload.get("target_client") != REACHY_CLIENT_ID:
            return
        self.seen_event_ids.add(event_id)
        self.reachy.perform(str(payload.get("emotion") or "calm"), _optional_string(payload.get("speech")))

    def _post_status(self, connection: ClientConnectionState, status: str, detail: str | None = None) -> None:
        self.http.post_json(
            f"/api/clients/{REACHY_CLIENT_ID}/status",
            {"connection": connection.value, "status": status, "detail": detail},
        )


def main(argv: list[str] | None = None) -> None:
    """Run the Reachy bridge."""
    args = _parse_args(argv)
    config = BridgeConfig(
        base_url=args.base_url,
        poll_seconds=args.poll_seconds,
        source=args.source,
        manual_confirm=args.manual_confirm,
        wake_on_connect=args.wake_on_connect,
    )
    adapter = _build_reachy_adapter(args)
    try:
        ReachyBridge(config=config, http=UrllibBridgeHttpClient(config.base_url), reachy=adapter).start()
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pokerbot-reachy-bridge", description="Run the Reachy Mini thin bridge.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Pokerbot app base URL.")
    parser.add_argument("--poll-seconds", default=DEFAULT_POLL_SECONDS, type=float, help="Polling interval.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Private-card frame source label.")
    parser.add_argument(
        "--reachy-daemon-url",
        help=(
            "Reachy Mini wireless daemon URL. When set, the bridge uses the daemon REST API instead of SDK/Zenoh."
        ),
    )
    parser.add_argument(
        "--reachy-connection-mode",
        choices=["auto", "localhost_only", "network"],
        default=DEFAULT_REACHY_CONNECTION_MODE,
        help="Reachy SDK connection mode.",
    )
    parser.add_argument(
        "--reachy-media-backend",
        default=DEFAULT_REACHY_MEDIA_BACKEND,
        help="Reachy SDK media backend.",
    )
    parser.add_argument(
        "--reachy-timeout",
        default=DEFAULT_REACHY_TIMEOUT_SECONDS,
        type=float,
        help="Reachy SDK connection timeout in seconds.",
    )
    capture_mode = parser.add_mutually_exclusive_group()
    capture_mode.add_argument("--auto-capture", action="store_true", help="Capture as soon as the orchestrator asks.")
    capture_mode.add_argument("--manual-confirm", action="store_true", help="Prompt before capturing private cards.")
    parser.add_argument(
        "--no-wake-on-connect",
        action="store_false",
        dest="wake_on_connect",
        help="Do not move Reachy into the default wake pose when the bridge connects.",
    )
    parser.add_argument("--console-only", action="store_true", help="Run without connecting to Reachy hardware.")
    return parser.parse_args(argv)


def _build_reachy_adapter(args: argparse.Namespace) -> ReachyAdapter:
    if args.console_only:
        return ConsoleReachyAdapter()
    if args.reachy_daemon_url:
        return ReachyDaemonHttpAdapter(str(args.reachy_daemon_url))
    return ReachyMiniAdapter(
        ReachySdkConfig(
            connection_mode=cast("ReachyConnectionMode", args.reachy_connection_mode),
            media_backend=args.reachy_media_backend,
            timeout_seconds=args.reachy_timeout,
        )
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


if __name__ == "__main__":
    main(sys.argv[1:])
