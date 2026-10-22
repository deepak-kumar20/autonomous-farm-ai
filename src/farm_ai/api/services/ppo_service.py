import logging
from typing import Optional, Dict, Any, Tuple, List

from ...environment.farm_environment import FarmEnvironment, _int_action_to_farm_action
from ...framework.agents import PPOPolicy, PPOCheckpointError
from ...framework.feasibility import FeasibilityFilter
from .farm_service import to_json_compatible

logger = logging.getLogger("farm_ai.api.ppo_service")


def action_details(action_idx: int) -> Dict[str, Any]:
    a = _int_action_to_farm_action(int(action_idx))
    return {"id": int(action_idx), "type": a.type.value, "label": str(a), "field_id": a.field_id, "crop_name": a.crop_name}


class PPOService:
    """Loads one PPO checkpoint at startup (after a compatibility check) and serves decisions.

    Decisions use the same Stage 2 → 3 path as experiments: the policy ranks all actions and the
    session's feasibility filter picks the first acceptable one.
    """

    def __init__(self):
        self._agent: Optional[PPOPolicy] = None
        self._model_path: Optional[str] = None
        self._is_loaded: bool = False
        self._load_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._agent.meta) if self._agent is not None else {}

    def load_model(self, model_path: str) -> bool:
        self._model_path = model_path
        try:
            self._agent = PPOPolicy(model_path, deterministic=True)
            self._is_loaded = True
            self._load_error = None
            logger.info("PPO model loaded from '%s'", self._agent.path)
            return True
        except PPOCheckpointError as e:
            self._load_error = f"PPO checkpoint unusable: {e}"
        except Exception as e:  # pragma: no cover - defensive
            self._load_error = f"Failed to load PPO model: {e}"
        logger.warning(self._load_error)
        self._is_loaded = False
        self._agent = None
        return False

    def _require(self) -> PPOPolicy:
        if not self._is_loaded or self._agent is None:
            raise RuntimeError(self._load_error or "PPO model is not loaded. Train one with scripts/train_ppo.py")
        return self._agent

    def ranked(self, env: FarmEnvironment, deterministic: bool = True) -> List[int]:
        agent = self._require()
        agent.deterministic = deterministic
        return agent.rank(env)

    def predict(self, env: FarmEnvironment, filt: Optional[FeasibilityFilter] = None,
                deterministic: bool = True) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        """Returns the action PPO would take now (after the filter), without side effects."""
        ranked = self.ranked(env, deterministic)
        raw = int(ranked[0])
        chosen, rejected = raw, 0
        if filt is not None and filt.mode != "none":
            for a in ranked:
                if filt.check(env, int(a)).accepted:
                    chosen = int(a)
                    break
                rejected += 1
        extra = {"policy_top_action": action_details(raw), "filter_rejections": rejected}
        return chosen, to_json_compatible(action_details(chosen)), to_json_compatible(extra)


# Global singleton instance for the FastAPI application
ppo_service = PPOService()
