from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional

from ..dependencies import get_farm_service, get_match_store, FarmSessionManager, MatchStore
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
from ..services.match_service import public_scenario
from ...framework.config import SimulationConfig

router = APIRouter(prefix="/farm", tags=["Farm Simulation"])


def config_from_request(req: ResetRequest) -> Optional[SimulationConfig]:
    fields = {k: getattr(req, k) for k in ("decision_order", "filter_mode", "threshold_rule", "completion_threshold", "market_model")}
    fields = {k: v for k, v in fields.items() if v is not None}
    if not fields:
        return None
    try:
        return SimulationConfig(**fields)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))


def _not_found(session_id: str):
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Farm session '{session_id}' not found")


@router.post("/reset", response_model=ResetResponse, summary="Create or Reset a Farm Session")
def reset_farm(
    request: Optional[ResetRequest] = None,
    service: FarmSessionManager = Depends(get_farm_service),
):
    """Creates a session (or resets `session_id`) on the scenario given by `profile` + `seed`.

    The same profile and seed always produce the same 30-day scenario. Optional fields choose the
    decision order, feasibility-filter mode and market model (defaults: LFM, deterministic, exogenous).
    """
    req = request or ResetRequest()
    try:
        session_id, state, info = service.reset_session(
            session_id=req.session_id, seed=req.seed, profile=req.profile, config=config_from_request(req),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    return ResetResponse(session_id=session_id, state=state, info=info)


@router.get("/{session_id}/state", response_model=FarmStateResponse, summary="Get Current Farm State")
def get_farm_state(session_id: str, service: FarmSessionManager = Depends(get_farm_service)):
    try:
        return FarmStateResponse(session_id=session_id, state=service.get_state(session_id))
    except KeyError:
        raise _not_found(session_id)


@router.get("/{session_id}/scenario", summary="Get the session's scenario (ex-ante risks only)")
def get_session_scenario(session_id: str, service: FarmSessionManager = Depends(get_farm_service)):
    try:
        sess = service.get_session(session_id)
    except KeyError:
        raise _not_found(session_id)
    return {**public_scenario(sess.env.scenario), "config": sess.config.to_dict(), "match_id": sess.match_id}


@router.get("/{session_id}/log", summary="Daily transition log of the session so far")
def get_session_log(session_id: str, service: FarmSessionManager = Depends(get_farm_service)):
    try:
        env = service.get_env(session_id)
    except KeyError:
        raise _not_found(session_id)
    from ..services.farm_service import to_json_compatible
    return {"session_id": session_id, "records": to_json_compatible(env.transition_log)}


@router.get("/{session_id}/actions", response_model=AvailableActionsResponse, summary="Get Available Actions")
def get_available_actions(session_id: str, service: FarmSessionManager = Depends(get_farm_service)):
    """All 24 discrete actions; `valid` is the session filter's verdict, `reason` explains a rejection."""
    try:
        return AvailableActionsResponse(session_id=session_id, actions=service.get_actions(session_id))
    except KeyError:
        raise _not_found(session_id)


@router.post("/{session_id}/action", response_model=StepResponse, summary="Execute Human / Client Action")
def execute_player_action(
    session_id: str,
    request: ActionRequest,
    service: FarmSessionManager = Depends(get_farm_service),
    matches: MatchStore = Depends(get_match_store),
):
    """Submits one action (0–23).

    The session's feasibility filter checks it first. If rejected, `accepted` is false, `feasibility`
    explains why, and nothing changes (the day does not advance). If accepted, the simulation advances
    one day. When a Human-vs-AI match session finishes day 30, the human episode is saved.
    """
    try:
        accepted, feas, state, reward, terminated, truncated, info = service.execute_action(session_id, request.action)
    except KeyError:
        raise _not_found(session_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid action execution: {e}")
    if terminated:
        sess = service.get_session(session_id)
        if sess.match_id:
            matches.complete_human(sess.match_id, sess.env)
    return StepResponse(
        session_id=session_id, accepted=accepted, feasibility=feas, state=state,
        reward=reward, terminated=terminated, truncated=truncated, info=info,
    )


@router.get("/{session_id}/metrics", response_model=FarmMetricsResponse, summary="Get Farm Financial & Operational Metrics")
def get_farm_metrics(session_id: str, service: FarmSessionManager = Depends(get_farm_service)):
    try:
        return FarmMetricsResponse(**service.get_metrics(session_id))
    except KeyError:
        raise _not_found(session_id)


@router.delete("/{session_id}", summary="Delete Farm Session", status_code=status.HTTP_200_OK)
def delete_farm_session(session_id: str, service: FarmSessionManager = Depends(get_farm_service)):
    if not service.delete_session(session_id):
        raise _not_found(session_id)
    return {"status": "deleted", "session_id": session_id}
