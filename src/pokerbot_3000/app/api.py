"""FastAPI routes for the orchestrator skeleton."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

from pokerbot_3000.domain.models import (
    ClientId,
    ClientStatus,
    ClientStatusUpdate,
    ExternalInputResult,
    GameEvent,
    HumanActionInput,
    HumanTableTalkInput,
    OperatorControlResult,
    PrivateCardFrameInput,
    PrivateCardObservation,
    PublicBoardFrameInput,
    PublicGameState,
    RevealedCardsFrameInput,
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
    router.include_router(_create_stale_voice_router())
    router.include_router(_create_voice_router(runtime))
    router.include_router(_create_presentation_router(runtime))
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
        return await runtime.submit_human_action(request)

    @router.post("/inputs/human-table-talk", response_model=ExternalInputResult)
    async def submit_human_table_talk(request: HumanTableTalkInput) -> ExternalInputResult:
        """Consume human speech addressed to Reachy or Eliza without advancing the action."""
        return await runtime.submit_human_table_talk(request)

    @router.post("/clients/{agent_id}/private-cards", response_model=ExternalInputResult)
    async def record_client_private_cards(
        agent_id: str,
        observation: PrivateCardObservation,
    ) -> ExternalInputResult:
        """Consume private-card input from a thin Reachy or Eliza client."""
        try:
            result = await runtime.record_client_private_cards(agent_id, observation)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return result

    @router.post("/clients/{agent_id}/private-cards/frame", response_model=ExternalInputResult)
    async def submit_client_private_cards_frame(
        agent_id: str,
        frame_input: PrivateCardFrameInput,
    ) -> ExternalInputResult:
        """Consume a thin-client private-card image frame."""
        if PUBLIC_BOARD_FRAME_DATA_URI.fullmatch(frame_input.data_uri) is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Private-card frame must be a JPEG or PNG data URI.",
            )
        return await runtime.process_private_cards_frame(
            agent_id,
            ImageFrame(source=frame_input.source, data_uri=frame_input.data_uri),
        )

    @router.post("/clients/{client_id}/status", response_model=ClientStatus)
    async def update_client_status(client_id: ClientId, update: ClientStatusUpdate) -> ClientStatus:
        """Record a thin-client connection/status update."""
        client_status = orchestrator.update_client_status(client_id, update)
        await runtime.broadcaster.publish_snapshot()
        return client_status

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

    @router.get("/voice/eliza/{event_id}", response_class=Response)
    async def get_eliza_voice(event_id: str) -> Response:
        """Return generated ElevenLabs audio for one Eliza speech event."""
        try:
            audio = await runtime.synthesize_eliza_event(event_id)
        except ElevenLabsConfigurationError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except ElevenLabsClientError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return Response(content=audio, media_type="audio/mpeg")

    @router.get("/voice/reachy/{event_id}", response_class=Response)
    async def get_reachy_voice(event_id: str) -> Response:
        """Return generated ElevenLabs audio for one Reachy speech event."""
        try:
            audio = await runtime.synthesize_reachy_event(event_id)
        except ElevenLabsConfigurationError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except ElevenLabsClientError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return Response(content=audio, media_type="audio/mpeg")

    return router


def _create_presentation_router(runtime: DashboardRuntime) -> APIRouter:
    router = APIRouter()

    @router.post("/presentation/{event_id}/complete", response_model=list[GameEvent])
    async def complete_presentation(event_id: str) -> list[GameEvent]:
        """Mark one client presentation event complete and resume queued work."""
        return await runtime.complete_presentation(event_id)

    return router


def _create_stale_voice_router() -> APIRouter:
    router = APIRouter()

    @router.post("/voice/transcript")
    async def reject_stale_browser_transcript() -> None:
        """Reject transcript posts from stale dashboard JavaScript bundles."""
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Stale dashboard JavaScript called removed /api/voice/transcript endpoint. Restart and reload.",
        )

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

    @router.post("/vision/showdown/revealed-cards", response_model=ExternalInputResult)
    async def submit_revealed_cards_frame(frame_input: RevealedCardsFrameInput) -> ExternalInputResult:
        """Consume one browser-captured revealed-card seat frame."""
        if PUBLIC_BOARD_FRAME_DATA_URI.fullmatch(frame_input.data_uri) is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Revealed-card frame must be a JPEG or PNG data URI.",
            )
        return await runtime.process_revealed_cards_frame(
            frame_input.seat,
            ImageFrame(source=frame_input.source, data_uri=frame_input.data_uri),
        )

    return router
