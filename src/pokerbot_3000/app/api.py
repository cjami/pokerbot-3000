"""FastAPI routes for the orchestrator skeleton."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query, status

from pokerbot_3000.domain.models import (
    ExternalInputResult,
    GameEvent,
    HumanActionInput,
    OperatorControlResult,
    PrivateCardObservation,
    PublicGameState,
)

if TYPE_CHECKING:
    from pokerbot_3000.orchestrator import InMemoryOrchestrator


def create_api_router(orchestrator: InMemoryOrchestrator) -> APIRouter:
    """Create API routes backed by the provided orchestrator."""
    router = APIRouter(prefix="/api", tags=["orchestrator"])

    @router.get("/state", response_model=PublicGameState)
    async def get_state() -> PublicGameState:
        """Return the current public game snapshot."""
        return orchestrator.public_state()

    @router.get("/events", response_model=list[GameEvent])
    async def get_events(limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[GameEvent]:
        """Return recent orchestrator events."""
        return orchestrator.events(limit=limit)

    @router.post("/game/start", response_model=OperatorControlResult)
    async def start_game() -> OperatorControlResult:
        """Start a fresh demo hand from the operator dashboard."""
        return orchestrator.start_game()

    @router.post("/game/stop", response_model=OperatorControlResult)
    async def stop_game() -> OperatorControlResult:
        """Stop orchestration from the operator dashboard."""
        return orchestrator.stop_game()

    @router.post("/inputs/human-action", response_model=ExternalInputResult)
    async def submit_human_action(request: HumanActionInput) -> ExternalInputResult:
        """Consume human action input and advance the engine until blocked."""
        return orchestrator.submit_human_action(request)

    @router.post("/clients/{agent_id}/private-cards", response_model=ExternalInputResult)
    async def record_client_private_cards(
        agent_id: str,
        observation: PrivateCardObservation,
    ) -> ExternalInputResult:
        """Consume private-card input from a thin Reachy or Eliza client."""
        try:
            return orchestrator.record_client_private_cards(agent_id, observation)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return router
