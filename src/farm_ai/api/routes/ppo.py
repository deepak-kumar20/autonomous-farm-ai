from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional

from ..dependencies import (
    get_farm_service,
    get_match_store,
    FarmSessionManager,
    MatchStore,
    get_ppo_service,
    PPOService,
)
from ..schemas.ppo import (
    PPOPredictRequest,
    PPOActionResponse,
    PPOStepRequest,
    PPOStepResponse,
)
from ..services.ppo_service import action_details
from ...framework.feasibility import decide

router = APIRouter(prefix="/ppo", tags=["PPO Agent"])


def _require_loaded(ppo: PPOService):
    if not ppo.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ppo.load_error or "PPO model is unavailable or not yet loaded",
        )


@router.post("/predict", response_model=PPOActionResponse, summary="Predict Action with PPO Agent")
def predict_action(
    request: PPOPredictRequest,
    farm_mgr: FarmSessionManager = Depends(get_farm_service),
    ppo: PPOService = Depends(get_ppo_service),
):
    """Returns the action PPO would take now in this session, after the session's feasibility filter.

    Does NOT advance the environment and records nothing.
    """
    _require_loaded(ppo)
    try:
        sess = farm_mgr.get_session(request.session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Farm session '{request.session_id}' not found")
    if sess.env.season_over:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Season is over")
    action_idx, details, extra = ppo.predict(sess.env, sess.filt, deterministic=request.deterministic)
    return PPOActionResponse(
        session_id=request.session_id, action=action_idx, action_details=details,
        deterministic=request.deterministic, **extra,
    )


@router.post("/{session_id}/step", response_model=PPOStepResponse, summary="Take a Step with PPO Agent")
def step_ppo(
    session_id: str,
    request: Optional[PPOStepRequest] = None,
    farm_mgr: FarmSessionManager = Depends(get_farm_service),
    ppo: PPOService = Depends(get_ppo_service),
    matches: MatchStore = Depends(get_match_store),
):
    """PPO proposes actions in order of preference; the session's filter accepts the first feasible one,
    which is then executed (one simulated day). Rejections are logged on the session's day record."""
    req = request or PPOStepRequest()
    _require_loaded(ppo)
    try:
        sess = farm_mgr.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Farm session '{session_id}' not found")
    if sess.env.season_over:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Season is over")

    ranked = ppo.ranked(sess.env, req.deterministic)
    before = len(sess.env._candidates)
    action_idx = decide(sess.env, ranked, sess.filt)
    rejections = sum(1 for c in sess.env._candidates[before:] if not c.accepted)
    state, reward, terminated, truncated, info = farm_mgr.step_accepted(session_id, action_idx)
    if terminated and sess.match_id:
        matches.complete_human(sess.match_id, sess.env)
    return PPOStepResponse(
        session_id=session_id, action=action_idx, action_details=action_details(action_idx),
        state=state, reward=reward, terminated=terminated, truncated=truncated, info=info,
        filter_rejections=rejections,
    )
