"""FastAPI application factory for Pokerbot 3000."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from fastapi import FastAPI, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJECT_DIR: Final = Path(__file__).resolve().parents[3]
PACKAGE_DIR: Final = Path(__file__).resolve().parents[1]
TEMPLATE_DIR: Final = PACKAGE_DIR / "web" / "templates"
GENERATED_STATIC_DIR: Final = PROJECT_DIR / "build" / "web" / "static"

templates = Jinja2Templates(directory=TEMPLATE_DIR)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Pokerbot 3000")
    app.mount("/static", StaticFiles(directory=GENERATED_STATIC_DIR, check_dir=False), name="static")

    @app.get("/", response_class=Response)
    async def index(request: Request) -> Response:
        return templates.TemplateResponse(
            request=request,
            name="index.html.jinja",
            context={"app_name": "Pokerbot 3000"},
        )

    return app
