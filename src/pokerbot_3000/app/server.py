"""FastAPI application factory for Pokerbot 3000."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

from fastapi import FastAPI, HTTPException, Request, WebSocket, status
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pokerbot_3000.app.api import create_api_router
from pokerbot_3000.app.runtime import DashboardRuntime
from pokerbot_3000.domain.models import ClientId

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

PROJECT_DIR: Final = Path(__file__).resolve().parents[3]
PACKAGE_DIR: Final = Path(__file__).resolve().parents[1]
TEMPLATE_DIR: Final = PACKAGE_DIR / "web" / "templates"
GENERATED_STATIC_DIR: Final = PROJECT_DIR / "build" / "web" / "static"

templates = Jinja2Templates(directory=TEMPLATE_DIR)


def create_app(runtime: DashboardRuntime | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app_runtime = runtime or DashboardRuntime.create_default()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await app_runtime.startup()
        try:
            yield
        finally:
            await app_runtime.shutdown()

    app = FastAPI(title="Pokerbot 3000", lifespan=lifespan)
    orchestrator = app_runtime.orchestrator
    app.state.runtime = app_runtime
    app.state.orchestrator = orchestrator
    app.include_router(create_api_router(app_runtime))
    app.mount("/static", StaticFiles(directory=GENERATED_STATIC_DIR, check_dir=False), name="static")

    @app.websocket("/ws/events")
    async def events_websocket(websocket: WebSocket) -> None:
        await app_runtime.broadcaster.websocket_endpoint(websocket)

    @app.websocket("/ws/clients/{client_id}")
    async def client_events_websocket(websocket: WebSocket, client_id: str) -> None:
        try:
            parsed_client_id = ClientId(client_id)
        except ValueError:
            await websocket.close(code=1008)
            return
        await app_runtime.broadcaster.websocket_endpoint(
            websocket,
            snapshot_factory=lambda: app_runtime.client_snapshot(parsed_client_id),
        )

    @app.websocket("/ws/voice/human")
    async def human_voice_websocket(websocket: WebSocket) -> None:
        await app_runtime.browser_voice_websocket_endpoint(websocket)

    @app.get("/", response_class=Response)
    async def index(request: Request) -> Response:
        state = orchestrator.public_state()
        return templates.TemplateResponse(
            request=request,
            name="index.html.jinja",
            context={
                "app_name": "Pokerbot 3000",
                "client_statuses": orchestrator.client_statuses().values(),
                "events": orchestrator.events(limit=8),
                "players": sorted(state.players.items()),
                "private_states": orchestrator.private_states().values(),
                "state": state,
                "static_version": _static_version(),
                "voice_input": app_runtime.voice_status(),
            },
        )

    @app.get("/clients/{client_id}", response_class=Response)
    async def client_page(request: Request, client_id: str) -> Response:
        if client_id != ClientId.ELIZA:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown thin client page.")
        state = orchestrator.public_state()
        return templates.TemplateResponse(
            request=request,
            name="client_eliza.html.jinja",
            context={
                "app_name": "Eliza",
                "client_id": ClientId.ELIZA,
                "state": state,
                "static_version": _static_version(),
            },
        )

    return app


def _static_version() -> str:
    newest_mtime = 0
    for filename in ("app.js", "eliza.js", "styles.css"):
        path = GENERATED_STATIC_DIR / filename
        if path.exists():
            newest_mtime = max(newest_mtime, int(path.stat().st_mtime))
    return str(newest_mtime)
