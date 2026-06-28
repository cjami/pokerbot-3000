"""FastAPI application factory for Pokerbot 3000."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pokerbot_3000.app.api import create_api_router
from pokerbot_3000.orchestrator import InMemoryOrchestrator

PROJECT_DIR: Final = Path(__file__).resolve().parents[3]
PACKAGE_DIR: Final = Path(__file__).resolve().parents[1]
TEMPLATE_DIR: Final = PACKAGE_DIR / "web" / "templates"
GENERATED_STATIC_DIR: Final = PROJECT_DIR / "build" / "web" / "static"

templates = Jinja2Templates(directory=TEMPLATE_DIR)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Pokerbot 3000")
    orchestrator = InMemoryOrchestrator()
    app.state.orchestrator = orchestrator
    app.include_router(create_api_router(orchestrator))
    app.mount("/static", StaticFiles(directory=GENERATED_STATIC_DIR, check_dir=False), name="static")

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
            },
        )

    return app
