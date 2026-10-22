"""Scenario and Human-vs-AI match endpoints."""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ..dependencies import get_match_store, get_ppo_service, MatchStore, PPOService
from ..services.match_service import load_or_create_scenario, public_scenario
from ..services.farm_service import to_json_compatible
from ...framework.config import SimulationConfig
from ...framework.scenario import PROFILES

scenario_router = APIRouter(prefix="/scenarios", tags=["Scenarios"])
match_router = APIRouter(prefix="/matches", tags=["Human vs AI"])
agents_router = APIRouter(prefix="/agents", tags=["Agents"])


class ScenarioRequest(BaseModel):
    profile: str = Field("baseline", description="baseline, dry, pest or combined")
    seed: Optional[int] = Field(None, ge=0, description="Omit for a random seed")


@scenario_router.get("/profiles", summary="Scenario profiles")
def list_profiles():
    return [{"name": p.name, "description": p.description, "drought_base": p.drought_base,
             "pest_base": p.pest_base, "shock_base": p.shock_base} for p in PROFILES.values()]


@scenario_router.post("", summary="Generate (or re-create) a 30-day scenario")
def create_scenario(req: ScenarioRequest):
    if req.profile not in PROFILES:
        raise HTTPException(status_code=422, detail=f"Unknown profile '{req.profile}'")
    return public_scenario(load_or_create_scenario(profile=req.profile, seed=req.seed))


@scenario_router.get("/{scenario_id}", summary="Ex-ante view of a saved scenario")
def get_scenario(scenario_id: str):
    try:
        return public_scenario(load_or_create_scenario(scenario_id=scenario_id))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class MatchRequest(BaseModel):
    scenario_id: str
    player_label: str = Field("human", max_length=40, description="Free-text label for the human participant/session")
    decision_order: str = "LFM"
    filter_mode: str = "deterministic"
    threshold_rule: str = "fixed"
    completion_threshold: float = Field(0.8, ge=0, le=1)
    market_model: str = "exogenous"


class AIRunRequest(BaseModel):
    agent: str = Field("ppo", description="ppo, rule_based or random")
    deterministic: bool = True


@match_router.post("", summary="Start a Human-vs-AI match (creates the human's session)")
def create_match(req: MatchRequest, store: MatchStore = Depends(get_match_store)):
    try:
        config = SimulationConfig(
            decision_order=req.decision_order, filter_mode=req.filter_mode, threshold_rule=req.threshold_rule,
            completion_threshold=req.completion_threshold, market_model=req.market_model,
        )
        scenario = load_or_create_scenario(scenario_id=req.scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    match = store.create(scenario, config, req.player_label)
    return {"match_id": match["match_id"], "session_id": match["human"]["session_id"],
            "config": match["config"], "scenario": public_scenario(scenario)}


@match_router.get("", summary="List saved matches")
def list_matches(store: MatchStore = Depends(get_match_store)):
    return store.list()


@match_router.get("/{match_id}", summary="Full match record (human and AI episodes)")
def get_match(match_id: str, store: MatchStore = Depends(get_match_store)):
    try:
        return store.load(match_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found")


@match_router.post("/{match_id}/ai-run", summary="Replay the match scenario with an AI agent")
def run_ai(match_id: str, req: AIRunRequest, store: MatchStore = Depends(get_match_store),
           ppo: PPOService = Depends(get_ppo_service)):
    if req.agent == "ppo" and not ppo.is_loaded:
        raise HTTPException(status_code=503, detail=ppo.load_error or "PPO model is not available")
    try:
        run = store.run_ai(match_id, req.agent, req.deterministic, ppo_path=ppo.model_path)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found")
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return to_json_compatible(run)


@agents_router.get("", summary="Agents available for AI play and replay")
def available_agents(ppo: PPOService = Depends(get_ppo_service)):
    meta = ppo.metadata
    return [
        {"agent": "ppo", "available": ppo.is_loaded, "status": "trained" if ppo.is_loaded else ppo.load_error,
         "timesteps": meta.get("timesteps"), "validation": meta.get("final_validation")},
        {"agent": "rule_based", "available": True, "status": "hand-written rules"},
        {"agent": "random", "available": True, "status": "uniform over its candidate actions"},
    ]
