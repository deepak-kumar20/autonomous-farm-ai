import uuid
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from ...environment.farm_environment import FarmEnvironment
from ...environment.actions import Action
from ...framework.config import SimulationConfig
from ...framework.feasibility import FeasibilityFilter, FeasibilityResult
from ...framework.scenario import Scenario, generate_scenario


def to_json_compatible(obj: Any) -> Any:
    """Recursively converts NumPy data types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_compatible(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_compatible(v) for v in obj]
    return obj


@dataclass
class Session:
    env: FarmEnvironment
    config: SimulationConfig
    filt: FeasibilityFilter
    match_id: Optional[str] = None


class FarmSessionManager:
    """In-memory farm sessions keyed by UUID.

    Every session owns one FarmEnvironment, the SimulationConfig it was created with, and the
    feasibility filter derived from that config. All actions — human or AI — go through the same
    filter and the same environment step.
    """

    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────
    def _new_session(self, scenario: Scenario, config: SimulationConfig, match_id: Optional[str]) -> Session:
        env = FarmEnvironment(config=config)
        env.reset(options={"scenario": scenario})
        return Session(env, config, FeasibilityFilter(config.filter_mode, config.completion_threshold, config.threshold_rule), match_id)

    def create_session(
        self,
        seed: Optional[int] = None,
        scenario: Optional[Scenario] = None,
        config: Optional[SimulationConfig] = None,
        profile: str = "baseline",
        match_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        session_id = str(uuid.uuid4())
        scenario = scenario or generate_scenario(profile, seed if seed is not None else int(np.random.SeedSequence().entropy % (2**31)))
        sess = self._new_session(scenario, config or SimulationConfig(), match_id)
        with self._lock:
            self._sessions[session_id] = sess
        return session_id, self.get_state(session_id), self._info(sess)

    def reset_session(
        self,
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
        profile: str = "baseline",
        config: Optional[SimulationConfig] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        if not session_id:
            return self.create_session(seed=seed, profile=profile, config=config)
        with self._lock:
            old = self._sessions.get(session_id)
        cfg = config or (old.config if old else SimulationConfig())
        scenario = generate_scenario(profile, seed if seed is not None else int(np.random.SeedSequence().entropy % (2**31)))
        sess = self._new_session(scenario, cfg, None)
        with self._lock:
            self._sessions[session_id] = sess
        return session_id, self.get_state(session_id), self._info(sess)

    @staticmethod
    def _info(sess: Session) -> Dict[str, Any]:
        return {
            "scenario_id": sess.env.scenario.scenario_id,
            "config": sess.config.to_dict(),
            "match_id": sess.match_id,
        }

    def get_session(self, session_id: str) -> Session:
        with self._lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"Session '{session_id}' not found")
        return sess

    def get_env(self, session_id: str) -> FarmEnvironment:
        return self.get_session(session_id).env

    # ── queries ───────────────────────────────────────────────────────
    def get_state(self, session_id: str) -> Dict[str, Any]:
        return to_json_compatible(self.get_env(session_id).get_dict_state())

    def get_actions(self, session_id: str) -> List[Dict[str, Any]]:
        sess = self.get_session(session_id)
        return to_json_compatible(sess.env.get_actions(sess.filt))

    def get_metrics(self, session_id: str) -> Dict[str, Any]:
        metrics = self.get_env(session_id).get_metrics()
        metrics["session_id"] = session_id
        return to_json_compatible(metrics)

    # ── actions ───────────────────────────────────────────────────────
    def execute_action(
        self, session_id: str, action: Union[int, Action]
    ) -> Tuple[bool, Dict[str, Any], Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Stage 3 → 6 for one proposed action.

        Returns (accepted, feasibility, state, reward, terminated, truncated, info). A rejected action
        changes nothing except the day's rejection log; the day does not advance.
        """
        sess = self.get_session(session_id)
        env = sess.env
        if env.season_over:
            raise ValueError("Season is over: start a new season to keep playing.")
        res: FeasibilityResult = sess.filt.check(env, int(action))
        env.record_candidate(res)
        if not res.accepted:
            return False, to_json_compatible(res.to_dict()), self.get_state(session_id), 0.0, False, False, {}
        _, reward, terminated, truncated, info = env.step(int(action))
        return (True, to_json_compatible(res.to_dict()), self.get_state(session_id),
                float(reward), bool(terminated), bool(truncated), to_json_compatible(info))

    def step_accepted(self, session_id: str, action: int):
        """Executes an action that a caller has already passed through the filter (e.g. framework.decide)."""
        env = self.get_env(session_id)
        _, reward, terminated, truncated, info = env.step(int(action))
        return self.get_state(session_id), float(reward), bool(terminated), bool(truncated), to_json_compatible(info)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Global singleton instance for the FastAPI application
farm_service = FarmSessionManager()
