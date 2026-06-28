"""FastAPI routes for the orchestrator skeleton."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from pokerbot_3000.domain.models import (
    ExternalInputResult,
    GameEvent,
    HumanActionInput,
    OperatorControlResult,
    PrivateCardObservation,
    PublicBoardFrameInput,
    PublicGameState,
)
from pokerbot_3000.ports.llm import ImageFrame
from pokerbot_3000.voice import ElevenLabsClientError, ElevenLabsConfigurationError

if TYPE_CHECKING:
    from pokerbot_3000.app.runtime import DashboardRuntime

PUBLIC_BOARD_FRAME_DATA_URI = re.compile(r"^data:image/(?:jpeg|png);base64,[A-Za-z0-9+/=]+$")


def create_api_router(runtime: DashboardRuntime) -> APIRouter:
    """Create API routes backed by the provided orchestrator."""
    router = APIRouter(prefix="/api", tags=["orchestrator"])
    router.include_router(_create_state_router(runtime))
    router.include_router(_create_game_router(runtime))
    router.include_router(_create_input_router(runtime))
    router.include_router(_create_vision_router(runtime))
    router.include_router(_create_voice_router(runtime))
    return router


def _create_state_router(runtime: DashboardRuntime) -> APIRouter:
    router = APIRouter()
    orchestrator = runtime.orchestrator

    @router.get("/state", response_model=PublicGameState)
    async def get_state() -> PublicGameState:
        """Return the current public game snapshot."""
        return orchestrator.public_state()

    @router.get("/events", response_model=list[GameEvent])
    async def get_events(limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[GameEvent]:
        """Return recent orchestrator events."""
        return orchestrator.events(limit=limit)

    return router


def _create_game_router(runtime: DashboardRuntime) -> APIRouter:
    router = APIRouter()

    @router.post("/game/start", response_model=OperatorControlResult)
    async def start_game() -> OperatorControlResult:
        """Start a fresh demo hand from the operator dashboard."""
        return await runtime.start_game()

    @router.post("/game/stop", response_model=OperatorControlResult)
    async def stop_game() -> OperatorControlResult:
        """Stop orchestration from the operator dashboard."""
        return await runtime.stop_game()

    return router


def _create_input_router(runtime: DashboardRuntime) -> APIRouter:
    router = APIRouter()
    orchestrator = runtime.orchestrator

    @router.post("/inputs/human-action", response_model=ExternalInputResult)
    async def submit_human_action(request: HumanActionInput) -> ExternalInputResult:
        """Consume human action input and advance the engine until blocked."""
        result = orchestrator.submit_human_action(request)
        await runtime.handle_new_events(result.events)
        await runtime.broadcaster.publish_snapshot()
        return result

    @router.post("/clients/{agent_id}/private-cards", response_model=ExternalInputResult)
    async def record_client_private_cards(
        agent_id: str,
        observation: PrivateCardObservation,
    ) -> ExternalInputResult:
        """Consume private-card input from a thin Reachy or Eliza client."""
        try:
            result = orchestrator.record_client_private_cards(agent_id, observation)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        await runtime.handle_new_events(result.events)
        await runtime.broadcaster.publish_snapshot()
        return result

    return router


def _create_voice_router(runtime: DashboardRuntime) -> APIRouter:
    router = APIRouter()

    @router.get("/voice/orchestrator/{event_id}", response_class=Response)
    async def get_orchestrator_voice(event_id: str) -> Response:
        """Return generated ElevenLabs audio for one orchestrator speech event."""
        try:
            audio = await runtime.synthesize_orchestrator_event(event_id)
        except ElevenLabsConfigurationError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except ElevenLabsClientError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return Response(content=audio, media_type="audio/mpeg")

    return router


def _create_vision_router(runtime: DashboardRuntime) -> APIRouter:
    router = APIRouter()

    @router.post("/vision/public-board/frame", response_model=ExternalInputResult)
    async def submit_public_board_frame(frame_input: PublicBoardFrameInput) -> ExternalInputResult:
        """Consume one browser-captured public-board frame."""
        if PUBLIC_BOARD_FRAME_DATA_URI.fullmatch(frame_input.data_uri) is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Public board frame must be a JPEG or PNG data URI.",
            )
        return await runtime.process_public_board_frame(
            ImageFrame(source=frame_input.source, data_uri=frame_input.data_uri),
        )

    return router
