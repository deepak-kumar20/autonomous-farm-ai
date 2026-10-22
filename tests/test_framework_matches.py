import hashlib
import json
import os
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from farm_ai.api.main import app  # noqa: E402


def _hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestMatches(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["FARM_DATA_DIR"] = self.tmp.name
        self.cm = TestClient(app)
        self.client = self.cm.__enter__()

    def tearDown(self):
        self.cm.__exit__(None, None, None)
        os.environ.pop("FARM_DATA_DIR", None)
        self.tmp.cleanup()

    def _play_human(self, session_id):
        """A simple scripted 'human': plant wheat on field 0 when empty, harvest when ready, then sell."""
        for _ in range(60):
            actions = {a["id"]: a for a in self.client.get(f"/api/farm/{session_id}/actions").json()["actions"]}
            choice = next((i for i in (1, 21, 6) if actions[i]["valid"]), 0)
            r = self.client.post(f"/api/farm/{session_id}/action", json={"action": choice}).json()
            if r["terminated"]:
                return
        self.fail("season did not finish")

    def test_match_flow_human_then_ai(self):
        sc = self.client.post("/api/scenarios", json={"profile": "pest", "seed": 11}).json()
        self.assertEqual(sc["scenario_id"], "S-pest-11")
        self.assertEqual(len(sc["timeline"]), 30)
        self.assertNotIn("pest_event", json.dumps(sc))          # realised events are not revealed
        m = self.client.post("/api/matches", json={"scenario_id": "S-pest-11", "player_label": "test"}).json()
        early = self.client.post(f"/api/matches/{m['match_id']}/ai-run", json={"agent": "rule_based"})
        self.assertEqual(early.status_code, 409)                 # human must finish first

        self._play_human(m["session_id"])
        rec = self.client.get(f"/api/matches/{m['match_id']}").json()
        self.assertEqual(rec["status"], "human_complete")
        self.assertEqual(len(rec["human"]["records"]), 30)

        run = self.client.post(f"/api/matches/{m['match_id']}/ai-run", json={"agent": "rule_based"}).json()
        human_ex = [(r["drought_event"], r["pest_event"], r["price_shock_event"]) for r in rec["human"]["records"]]
        ai_ex = [(r["drought_event"], r["pest_event"], r["price_shock_event"]) for r in run["records"]]
        self.assertEqual(human_ex, ai_ex)                          # identical exogenous sequence
        self.assertEqual(run["summary"]["initial_wealth"], rec["human"]["summary"]["initial_wealth"])
        self.assertEqual(run["summary"]["scenario_id"], rec["human"]["summary"]["scenario_id"])

    def test_human_run_does_not_mutate_scenario(self):
        self.client.post("/api/scenarios", json={"profile": "combined", "seed": 3})
        path = os.path.join(self.tmp.name, "scenarios", "S-combined-3.json")
        before = _hash(path)
        m = self.client.post("/api/matches", json={"scenario_id": "S-combined-3"}).json()
        self._play_human(m["session_id"])
        self.assertEqual(_hash(path), before)
        self.client.post(f"/api/matches/{m['match_id']}/ai-run", json={"agent": "random"})
        self.assertEqual(_hash(path), before)

    def test_human_episode_replayable(self):
        """Replaying the saved human action list on a fresh session reproduces the saved outcome."""
        m = self.client.post("/api/matches", json={"scenario_id": "S-baseline-21"}).json()
        self._play_human(m["session_id"])
        rec = self.client.get(f"/api/matches/{m['match_id']}").json()
        sid = self.client.post("/api/farm/reset", json={"profile": "baseline", "seed": 21}).json()["session_id"]
        for r in rec["human"]["records"]:
            self.client.post(f"/api/farm/{sid}/action", json={"action": r["action_id"]})
        metrics = self.client.get(f"/api/farm/{sid}/metrics").json()
        self.assertAlmostEqual(metrics["total_wealth"], rec["human"]["summary"]["final_wealth"], places=6)


if __name__ == "__main__":
    unittest.main()
