from .health import router as health_router
from .farm import router as farm_router
from .ppo import router as ppo_router
from .matches import scenario_router, match_router, agents_router

__all__ = ["health_router", "farm_router", "ppo_router", "scenario_router", "match_router", "agents_router"]
