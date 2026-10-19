from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional

from ..dependencies import get_farm_service, FarmSessionManager
from ..schemas.farm import (
    ResetRequest,
    ResetResponse,
    FarmStateResponse,
    FarmMetricsResponse,
)
from ..schemas.actions import (
    AvailableActionsResponse,
    ActionRequest,
    StepResponse,
)

router = APIRouter(prefix="/farm", tags=["Farm Simulation"])


@router.post(
    "/reset",
    response_model=ResetResponse,
    summary="Create or Reset a Farm Session",
    status_code=status.HTTP_200_OK,
)
def reset_farm(
    request: Optional[ResetRequest] = None,
    service: FarmSessionManager = Depends(get_farm_service),
):
    """Creates a new simulation session or resets an existing session.

    If `session_id` is provided in the request body, that session is reset.
    Otherwise, a new session is created with a unique UUID.
    Optional `seed` parameter controls environmental reproducibility.
    """
    req = request or ResetRequest()
    session_id, state, info = service.reset_session(
        session_id=req.session_id,
        seed=req.seed,
    )
    return ResetResponse(
        session_id=session_id,
        state=state,
        info=info,
    )


@router.get(
    "/{session_id}/state",
    response_model=FarmStateResponse,
    summary="Get Current Farm State",
)
def get_farm_state(
    session_id: str,
    service: FarmSessionManager = Depends(get_farm_service),
):
    """Retrieves the complete human-readable state of the farm for the given session."""
    try:
        state = service.get_state(session_id)
        return FarmStateResponse(session_id=session_id, state=state)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Farm session '{session_id}' not found",
        )


@router.get(
    "/{session_id}/actions",
    response_model=AvailableActionsResponse,
    summary="Get Available Actions",
)
def get_available_actions(
    session_id: str,
    service: FarmSessionManager = Depends(get_farm_service),
):
    """Returns all 24 discrete actions with dynamic validity checks against the current farm state."""
    try:
        actions = service.get_actions(session_id)
        return AvailableActionsResponse(session_id=session_id, actions=actions)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Farm session '{session_id}' not found",
        )


@router.post(
    "/{session_id}/action",
    response_model=StepResponse,
    summary="Execute Human / Client Action",
)
def execute_player_action(
    session_id: str,
    request: ActionRequest,
    service: FarmSessionManager = Depends(get_farm_service),
):
    """Executes a discrete action (0-23) in the simulation.

    Advances the simulation by one day, updates crop growth, market prices,
    risk events, and computes the wealth-shaped reward.
    """
    try:
        state, reward, terminated, truncated, info = service.execute_action(
            session_id=session_id,
            action=request.action,
        )
        return StepResponse(
            session_id=session_id,
            state=state,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Farm session '{session_id}' not found",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action execution: {str(e)}",
        )


@router.get(
    "/{session_id}/metrics",
    response_model=FarmMetricsResponse,
    summary="Get Farm Financial & Operational Metrics",
)
def get_farm_metrics(
    session_id: str,
    service: FarmSessionManager = Depends(get_farm_service),
):
    """Retrieves economic and operational metrics (cash, wealth, profit, survival status) calculated by the simulation."""
    try:
        metrics = service.get_metrics(session_id)
        return FarmMetricsResponse(**metrics)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Farm session '{session_id}' not found",
        )


@router.delete(
    "/{session_id}",
    summary="Delete Farm Session",
    status_code=status.HTTP_200_OK,
)
def delete_farm_session(
    session_id: str,
    service: FarmSessionManager = Depends(get_farm_service),
):
    """Deletes an active session to free in-memory resources."""
    success = service.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Farm session '{session_id}' not found",
        )
    return {"status": "deleted", "session_id": session_id}
