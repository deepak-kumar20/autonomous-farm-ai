"""Builds research tables, charts, diagrams and the generated documents from the CSV datasets.

Every number in a table/chart is computed here from research_outputs/data/*.csv (or logs/*.csv).
Missing or empty inputs produce an explicit "Not run" row or a chart status note — never a made-up value.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from . import stats as st
from .mapping import IMPLEMENTATION_MAPPING
from ..config.settings import CROP_CONFIG, NUM_FIELDS, SEASON_LENGTH, STARTING_CASH
from ..framework.scenario import PROFILES, RISK_LABEL_THRESHOLDS
from ..framework.config import SimulationConfig, PriceImpactParams

AGENT_LABEL = {"random": "Random", "rule_based": "Rule-based", "ppo": "PPO", "human": "Human"}
AGENT_COLOR = {"random": "#2a78d6", "rule_based": "#eb6834", "ppo": "#1baf7a", "human": "#eda100"}
CONFIG_LABEL = {
    "full": "Full framework (LFM + deterministic filter)",
    "no_filter": "No feasibility filter",
    "order_fml": "FML order (losses after decision)",
    "uncertainty_fixed": "Uncertainty filter (fixed τ = 0.8)",
    "uncertainty_break_even": "Uncertainty filter (break-even τ)",
    "price_impact": "Price-impact market",
}
FILTER_COLOR = {"no_filter": "#2a78d6", "full": "#eb6834", "uncertainty_fixed": "#1baf7a", "uncertainty_break_even": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PROFILE_ORDER = ["baseline", "dry", "pest", "combined"]


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────
def _read(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, low_memory=False)
    return df if len(df) else None


def _ok(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Valid completed episodes only (failed/incomplete runs excluded and reported elsewhere)."""
    if df is None:
        return None
    m = df["error"].isna() & (df["season_completed"].astype(str) == "True")
    return df[m]


def _fmt(x, kind="money"):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if kind == "money":
        return f"{x:,.0f}"
    if kind == "rate":
        return f"{x:.3f}"
    if kind == "pct":
        return f"{100 * x:.1f}%"
    if kind == "num":
        return f"{x:,.2f}"
    return str(x)


def _write_table(tables_dir: str, name: str, df: pd.DataFrame, title: str, notes: List[str]) -> None:
    assert df.shape[1] <= 10, f"{name} has {df.shape[1]} columns"
    df.to_csv(os.path.join(tables_dir, f"{name}.csv"), index=False)
    lines = [f"# {title}", ""]
    cols = list(df.columns)
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    if notes:
        lines += [""] + [f"*{n}*" if not n.startswith("-") else n for n in notes]
    with open(os.path.join(tables_dir, f"{name}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _rate(num: float, den: float) -> Optional[float]:
    return (num / den) if den else None


def _group_stats(df: pd.DataFrame) -> Dict[str, Any]:
    d = st.describe(df["profit"].tolist())
    planted = df["crops_planted"].sum()
    return {
        "n": d["n"],
        "mean_profit": d["mean"], "sd_profit": d["sd"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
        "mean_final_wealth": df["final_wealth"].mean(),
        "mean_harvest": df["harvest_qty"].mean(),
        "mean_destroyed": df["crops_destroyed"].mean(),
        "mean_drought_days": df["drought_setback_days"].mean(),
        "invalid_rate": _rate(df["invalid_actions"].sum(), df["actions_attempted"].sum()),
        "mean_rejections": df["feasibility_rejections"].mean(),
        "rejection_rate": _rate(df["feasibility_rejections"].sum(), df["candidates_evaluated"].sum()),
        "completion_rate": _rate(df["crops_harvested"].sum(), planted),
        "decision_ms": 1000 * df["decision_time_s"].mean(),
        "filter_ms": df["filter_time_ms"].mean(),
        "candidates": int(df["candidates_evaluated"].sum()),
        "rejected": int(df["feasibility_rejections"].sum()),
    }


def _paired(df: pd.DataFrame, a_mask, b_mask, key="scenario_seed") -> Dict[str, Any]:
    a = dict(zip(df[a_mask][key], df[a_mask]["profit"]))
    b = dict(zip(df[b_mask][key], df[b_mask]["profit"]))
    return st.paired(a, b)


def _fmt_paired(p: Dict[str, Any]) -> str:
    if not p or p.get("n_pairs", 0) == 0:
        return "—"
    if p["ci_low"] is None:
        return f"{p['mean']:+,.0f} (n={p['n_pairs']})"
    return f"{p['mean']:+,.0f} [{p['ci_low']:+,.0f}, {p['ci_high']:+,.0f}]"


# ──────────────────────────────────────────────────────────────────────────
# tables
# ──────────────────────────────────────────────────────────────────────────
def build_tables(out: str, cfg: Dict[str, Any], status: Dict[str, Any]) -> Dict[str, Any]:
    data, tables = os.path.join(out, "data"), os.path.join(out, "tables")
    os.makedirs(tables, exist_ok=True)
    summary = _ok(_read(os.path.join(data, "episode_summary.csv")))
    raw_summary = _read(os.path.join(data, "episode_summary.csv"))
    ledger = _read(os.path.join(data, "crop_ledger.csv"))
    results: Dict[str, Any] = {}
    n = cfg["n_scenarios"] if status.get("n_scenarios") is None else status["n_scenarios"]
    default = SimulationConfig()
    pi = PriceImpactParams()

    # ── Table 1 ───────────────────────────────────────────────────────
    rows = [
        ("Season length", "Days per season", SEASON_LENGTH, "days", "config/settings.py SEASON_LENGTH", "Actions on days 1–30; terminal valuation at the day-30 price"),
        ("Decision interval", "One action per simulated day", 1, "day", "FarmEnvironment.step", "No hourly steps"),
        ("Fields", "Number of plots", NUM_FIELDS, "fields", "config/settings.py NUM_FIELDS", ""),
        ("Initial cash", "Starting cash = initial wealth", f"{STARTING_CASH:,.0f}", "₹", "config/settings.py STARTING_CASH", "No debt or credit modelled"),
    ]
    for c, v in CROP_CONFIG.items():
        rows.append((f"{c}", "Seed cost / growth / yield / base price",
                     f"{v['planting_cost']:,.0f} / {v['growth_days']} / {v['yield_amount']} / {v['base_price']:g}",
                     "₹ / days / units / ₹ per unit", "config/settings.py CROP_CONFIG", "Units are abstract produce units, not kg"))
    for p in PROFILES.values():
        rows.append((f"Profile: {p.name}", "Base daily probabilities: drought / pest / price shock",
                     f"{p.drought_base:g} / {p.pest_base:g} / {p.shock_base:g}", "probability per day",
                     "framework/scenario.py PROFILES", "Daily risk = base × lognormal AR(1) multiplier"))
    pb = PROFILES["baseline"]
    rows += [
        ("Risk process", "Volatility σ / persistence ρ of daily risk multiplier", f"{pb.risk_volatility:g} / {pb.risk_persistence:g}", "—", "ScenarioProfile", "Mean multiplier ≈ 1; risk capped at 0.9"),
        ("Risk labels", "LOW / MEDIUM / HIGH cut-offs", f"<{RISK_LABEL_THRESHOLDS[0]:g} / <{RISK_LABEL_THRESHOLDS[1]:g} / ≥{RISK_LABEL_THRESHOLDS[1]:g}", "probability per day", "framework/scenario.py", "Display only"),
        ("Price drift", "Uniform daily drift; clamp before shocks", f"±{pb.price_drift:.0%}; [{pb.price_floor:g}, {pb.price_cap:g}] × base", "fraction", "ScenarioProfile", "Original Market rule"),
        ("Price shock", "Multiplier on all crops on a shock day", f"{pb.shock_factors[0]:g} or {pb.shock_factors[1]:g}", "×", "ScenarioProfile", "Equiprobable"),
        ("Drought effect", "GROWING crops with progress > 0 lose one day", 1, "day", "FarmEnvironment._resolve_losses", "Mature crops unaffected"),
        ("Pest effect", "Destroys one planted field (uniform pick)", 1, "field", "FarmEnvironment._resolve_losses", "Only on outbreak days"),
        ("Decision order (default)", "Stage order within a day", default.decision_order, "—", "SimulationConfig", "LFM = Loss → Field → Market"),
        ("Feasibility filter (default)", "Pre-execution filter mode", default.filter_mode, "—", "SimulationConfig", "Ablations: none / uncertainty"),
        ("Uncertainty threshold", "Fixed τ, or break-even τ = cost / (yield × price)", f"{default.completion_threshold:g} or break-even", "probability", "SimulationConfig", ""),
        ("Market model (default)", "Price formation", default.market_model, "—", "SimulationConfig / models/market.py", f"price_impact: Q_ref={pi.reference_supply:g}, γ={pi.elasticity:g}, δ={pi.decay:g}"),
        ("Invalid-action penalty", "Reward when an executed action fails", -50, "₹ (reward units)", "environment/rewards.py", "Reward = Δ wealth otherwise"),
        ("Evaluation scenarios", "Seeds per profile", n, "scenarios", "experiment_config.json", f"Seeds {cfg.get('seed_offset', 0)}–{cfg.get('seed_offset', 0) + n - 1}"),
    ]
    _write_table(tables, "table_1_simulation_configuration",
                 pd.DataFrame(rows, columns=["Parameter", "Description", "Value", "Unit", "Configuration source", "Notes"]),
                 "Table 1 — Simulation configuration", [])

    # ── Table 2 ───────────────────────────────────────────────────────
    _write_table(tables, "table_2_framework_implementation_mapping",
                 pd.DataFrame(IMPLEMENTATION_MAPPING, columns=["Component", "Intended behavior", "Actual implementation", "Status", "Verification test", "Notes"]),
                 "Table 2 — Framework implementation mapping",
                 ["Verified = implemented and covered by the named passing test (python -m unittest discover -s tests)."])

    if summary is None:
        for name, title in [("table_3_agent_configurations", "Table 3"), ("table_5_main_agent_comparison", "Table 5"),
                            ("table_6_feasibility_ablation", "Table 6"), ("table_7_decision_order_ablation", "Table 7"),
                            ("table_9_environmental_robustness", "Table 9"), ("table_10_feasibility_performance", "Table 10")]:
            _write_table(tables, name, pd.DataFrame([{"Status": "Not run", "Reason": "No episode data (run scripts/run_experiments.py)"}]), title, [])
        return results

    def sel(exp, **kw):
        m = summary["experiments"].str.split(";").apply(lambda xs: exp in xs)
        for k, v in kw.items():
            m &= summary[k] == v
        return summary[m]

    agents = [a for a in ["random", "rule_based", "ppo"] if a in set(summary["agent"])]

    # ── Table 3 ───────────────────────────────────────────────────────
    ppo_meta = {}
    meta_path = os.path.join(os.path.dirname(out), (cfg.get("ppo_checkpoint") or "") + ".meta.json")
    if cfg.get("ppo_checkpoint") and os.path.isfile(meta_path):
        with open(meta_path) as f:
            ppo_meta = json.load(f)
    t3 = []
    for a in ["random", "rule_based", "ppo"]:
        A = sel("A_agent_comparison", agent=a)
        B = sel("B_feasibility_ablation", agent=a, config_name="no_filter", scenario_profile="baseline")
        ag_status = status.get("agents", {}).get(a, {})
        if a not in cfg.get("agents", []):
            t3.append({"Agent": AGENT_LABEL[a], "Policy type": "—", "Training status": "Not configured in this run",
                       "Checkpoint or configuration": "—", "Action space": "Discrete(24)", "Evaluation episodes": 0,
                       "Mean profit (₹, full framework)": "Not run", "Invalid-action rate without filter": "Not run"})
            continue
        if a == "ppo":
            ptype = "PPO, MLP 128-128, greedy (argmax) ranking"
            train = (f"Trained {ppo_meta.get('timesteps', '?'):,} steps (seed {ppo_meta.get('seed')})" if ppo_meta else "Not trained") \
                if ag_status.get("status") == "ok" else f"Skipped: {ag_status.get('reason')}"
            ck = (cfg.get("ppo_checkpoint") or "—") + ".zip"
        elif a == "random":
            ptype, train, ck = "Uniform over own candidate set", "None (no learning)", "seed = scenario seed"
        else:
            ptype, train, ck = "Hand-written priority rules", "None (no learning)", "agents/rule_based_agent.py"
        g = _group_stats(A) if len(A) else None
        gb = _group_stats(B) if len(B) else None
        t3.append({
            "Agent": AGENT_LABEL[a], "Policy type": ptype, "Training status": train, "Checkpoint or configuration": ck,
            "Action space": "Discrete(24), shared with human",
            "Evaluation episodes": g["n"] if g else 0,
            "Mean profit (₹, full framework)": _fmt(g["mean_profit"]) if g else "Not run",
            "Invalid-action rate without filter": _fmt(gb["invalid_rate"], "rate") if gb else "Not run",
        })
    _write_table(tables, "table_3_agent_configurations", pd.DataFrame(t3), "Table 3 — Agent configurations",
                 ["Mean profit from Experiment A (baseline scenarios, full framework). Invalid-action rate = invalid executed actions / actions attempted, from Experiment B's no-filter arm (baseline)."])

    # ── Table 4 ───────────────────────────────────────────────────────
    man = _read(os.path.join(data, "scenario_manifest.csv"))
    t4 = []
    for p in PROFILES.values():
        mm = man[man["profile"] == p.name] if man is not None else None
        obs = (f"; observed mean events/season: drought {mm['drought_events'].mean():.1f}, pest {mm['pest_events'].mean():.1f}, "
               f"shocks {mm['price_shocks'].mean():.1f}") if mm is not None and len(mm) else ""
        t4.append({"Scenario ID": f"S-{p.name}-<seed>", "Weather profile": "Binary drought event (no temperature/rain variables)",
                   "Drought setting": f"base {p.drought_base:.0%}/day", "Pest setting": f"base {p.pest_base:.0%}/day",
                   "Market setting": f"exogenous random walk, shocks {p.shock_base:.0%}/day",
                   "Number of days": SEASON_LENGTH, "Seed count": int(len(mm)) if mm is not None else 0,
                   "Description": p.description + obs})
    t4.append({"Scenario ID": "S-baseline-<seed> (market variant)", "Weather profile": "as baseline", "Drought setting": "as baseline",
               "Pest setting": "as baseline", "Market setting": f"price_impact (Q_ref {pi.reference_supply:g}, γ {pi.elasticity:g}, δ {pi.decay:g})",
               "Number of days": SEASON_LENGTH, "Seed count": int(len(sel("M_market_model", config_name="price_impact")["scenario_seed"].unique())),
               "Description": "Same exogenous draws as baseline; own sales depress the received price"})
    _write_table(tables, "table_4_environmental_scenarios", pd.DataFrame(t4), "Table 4 — Environmental scenarios",
                 ["Scenario groups share seeds and per-subsystem random streams (common random numbers), so the same seed differs across groups only in the changed rate."])

    # ── Table 5 ───────────────────────────────────────────────────────
    t5 = []
    for a in agents:
        g = _group_stats(sel("A_agent_comparison", agent=a))
        results.setdefault("A", {})[a] = g
        t5.append({"Agent": AGENT_LABEL[a], "Episodes": g["n"], "Mean final wealth (₹)": _fmt(g["mean_final_wealth"]),
                   "Mean profit (₹)": _fmt(g["mean_profit"]), "Profit SD (₹)": _fmt(g["sd_profit"]),
                   "Mean harvest (units)": _fmt(g["mean_harvest"], "num"), "Mean crops destroyed": _fmt(g["mean_destroyed"], "num"),
                   "Invalid-action rate": _fmt(g["invalid_rate"], "rate"), "Mean feasibility rejections": _fmt(g["mean_rejections"], "num"),
                   "95% CI of mean profit (₹)": st.fmt_ci(g["ci_low"], g["ci_high"])})
    pairs = []
    A = sel("A_agent_comparison")
    for x in agents:
        for y in agents:
            if x < y:
                p = _paired(A, A["agent"] == x, A["agent"] == y)
                results.setdefault("A_pairs", {})[f"{x}-{y}"] = p
                pairs.append(f"- Paired difference {AGENT_LABEL[x]} − {AGENT_LABEL[y]}: {_fmt_paired(p)} ₹ (mean [95% CI], n = {p['n_pairs']} scenarios; {x} higher in {_fmt(p['share_positive'], 'pct')} of scenarios)")
    _write_table(tables, "table_5_main_agent_comparison", pd.DataFrame(t5), "Table 5 — Main agent comparison (Experiment A, baseline scenarios, full framework)",
                 ["CI = mean ± t(0.975, n−1)·s/√n. Invalid-action rate = invalid / actions attempted (30 per episode). Rejections = filter rejections per episode."] + pairs)

    # ── Table 6 ───────────────────────────────────────────────────────
    t6, notes6 = [], ["Completion rate = crops harvested / crops planted. Decision time = mean wall-clock time per episode spent in candidate ranking + filtering."]
    B = sel("B_feasibility_ablation")
    for prof in [p for p in PROFILE_ORDER if p in set(B["scenario_profile"])]:
        for a in agents:
            for cn in ["full", "no_filter"]:
                g = _group_stats(B[(B["agent"] == a) & (B["config_name"] == cn) & (B["scenario_profile"] == prof)])
                results.setdefault("B", {})[(a, prof, cn)] = g
                t6.append({"Configuration": f"{AGENT_LABEL[a]} · {prof} · {'full' if cn == 'full' else 'no filter'}",
                           "Filter enabled": "Yes (deterministic)" if cn == "full" else "No", "Scenario count": g["n"],
                           "Mean final wealth (₹)": _fmt(g["mean_final_wealth"]), "Mean profit (₹)": _fmt(g["mean_profit"]),
                           "Invalid-action rate": _fmt(g["invalid_rate"], "rate"), "Mean rejections": _fmt(g["mean_rejections"], "num"),
                           "Crop completion rate": _fmt(g["completion_rate"], "pct"), "Decision time (ms/episode)": _fmt(g["decision_ms"], "num")})
            sub = B[(B["agent"] == a) & (B["scenario_profile"] == prof)]
            p = _paired(sub, sub["config_name"] == "full", sub["config_name"] == "no_filter")
            results.setdefault("B_pairs", {})[(a, prof)] = p
            notes6.append(f"- {AGENT_LABEL[a]}, {prof}: paired profit difference full − no filter = {_fmt_paired(p)} ₹")
    _write_table(tables, "table_6_feasibility_ablation", pd.DataFrame(t6), "Table 6 — Feasibility-filter ablation (Experiment B)", notes6)

    # ── Table 7 ───────────────────────────────────────────────────────
    t7, notes7 = [], ["Only the stage order changes; scenario, agent, filter (deterministic) and market are identical. Difference = paired (FML − LFM) profit, mean [95% CI]."]
    C = sel("C_decision_order")
    for prof in [p for p in PROFILE_ORDER if p in set(C["scenario_profile"])]:
        for a in agents:
            sub = C[(C["agent"] == a) & (C["scenario_profile"] == prof)]
            p = _paired(sub, sub["config_name"] == "order_fml", sub["config_name"] == "full")
            results.setdefault("C_pairs", {})[(a, prof)] = p
            for cn, order in [("full", "Loss → Field → Market (LFM)"), ("order_fml", "Field → Market → Loss (FML)")]:
                g = _group_stats(sub[sub["config_name"] == cn])
                t7.append({"Configuration": f"{AGENT_LABEL[a]} · {prof}", "Stage order": order, "Episodes": g["n"],
                           "Mean final wealth (₹)": _fmt(g["mean_final_wealth"]), "Mean profit (₹)": _fmt(g["mean_profit"]),
                           "Profit SD (₹)": _fmt(g["sd_profit"]), "Mean crops destroyed": _fmt(g["mean_destroyed"], "num"),
                           "Invalid-action rate": _fmt(g["invalid_rate"], "rate"),
                           "Difference from full framework (₹)": "reference" if cn == "full" else _fmt_paired(p)})
    _write_table(tables, "table_7_decision_order_ablation", pd.DataFrame(t7), "Table 7 — Decision-order ablation (Experiment C)", notes7)

    # ── Table 8 ───────────────────────────────────────────────────────
    hv = _read(os.path.join(data, "human_ai_comparison_raw.csv"))
    if hv is None or "human" not in set(hv.get("player_type", [])):
        t8 = pd.DataFrame([{"Scenario ID": "—", "Player type": "Not run", "Initial wealth (₹)": "—", "Final wealth (₹)": "—",
                            "Profit (₹)": "—", "Harvest (units)": "—", "Crops destroyed": "—", "Invalid actions": "—",
                            "Feasibility rejections": "—", "Season completion": "No completed human sessions recorded yet"}])
        _write_table(tables, "table_8_human_vs_ai", t8, "Table 8 — Human vs AI (Experiment E)",
                     ["Play a Human-vs-AI match in the app (Versus page), then rerun scripts/run_experiments.py. No human results are fabricated."])
        results["E"] = None
    else:
        t8 = []
        for _, r in hv.iterrows():
            t8.append({"Scenario ID": r["scenario_id"], "Player type": "Human" if r["player_type"] == "human" else f"AI ({AGENT_LABEL.get(r['player'], r['player'])})",
                       "Initial wealth (₹)": _fmt(r["initial_wealth"]), "Final wealth (₹)": _fmt(r["final_wealth"]), "Profit (₹)": _fmt(r["profit"]),
                       "Harvest (units)": int(r["harvest_qty"]), "Crops destroyed": int(r["crops_destroyed"]), "Invalid actions": int(r["invalid_actions"]),
                       "Feasibility rejections": int(r["feasibility_rejections"]), "Season completion": "Yes" if str(r["season_completed"]) == "True" else "No"})
        notes8 = []
        hum = hv[hv["player_type"] == "human"]
        nm = hum["match_id"].nunique()
        for a in agents:
            ai = hv[(hv["player_type"] == "ai") & (hv["player"] == a)]
            p = st.paired(dict(zip(ai["match_id"], ai["profit"])), dict(zip(hum["match_id"], hum["profit"])))
            results.setdefault("E", {})[a] = p
            notes8.append(f"- {AGENT_LABEL[a]} − Human profit: {_fmt_paired(p)} ₹ over {p['n_pairs']} matched scenario(s); positive favours the AI.")
        label = "case study (single human session)" if nm == 1 else f"{nm} human sessions"
        _write_table(tables, "table_8_human_vs_ai", pd.DataFrame(t8), f"Table 8 — Human vs AI (Experiment E, {label})", notes8)

    # ── Table 9 ───────────────────────────────────────────────────────
    D = sel("D_environmental_robustness")
    t9 = []
    for prof in [p for p in PROFILE_ORDER if p in set(D["scenario_profile"])]:
        for a in agents:
            sub = D[(D["agent"] == a) & (D["scenario_profile"] == prof)]
            g = _group_stats(sub)
            base = D[(D["agent"] == a) & (D["scenario_profile"] == "baseline")]
            p = st.paired(dict(zip(sub["scenario_seed"], sub["profit"])), dict(zip(base["scenario_seed"], base["profit"])))
            results.setdefault("D", {})[(a, prof)] = {**g, "diff": p}
            t9.append({"Scenario": prof, "Agent": AGENT_LABEL[a], "Episodes": g["n"], "Mean profit (₹)": _fmt(g["mean_profit"]),
                       "Profit SD (₹)": _fmt(g["sd_profit"]), "Mean final wealth (₹)": _fmt(g["mean_final_wealth"]),
                       "Mean harvest (units)": _fmt(g["mean_harvest"], "num"), "Mean crops destroyed": _fmt(g["mean_destroyed"], "num"),
                       "Invalid-action rate": _fmt(g["invalid_rate"], "rate"),
                       "Difference from baseline (₹)": "reference" if prof == "baseline" else _fmt_paired(p)})
    _write_table(tables, "table_9_environmental_robustness", pd.DataFrame(t9), "Table 9 — Environmental robustness (Experiment D, full framework)",
                 ["Difference = paired (stress − baseline) profit for the same agent and seed, mean [95% CI]."])

    # ── Table 10 ──────────────────────────────────────────────────────
    F = sel("F_feasibility_filters")
    t10 = []
    notes10 = ["Rejection rate = rejected candidates / candidates evaluated by the filter (not / days). "
               "Completion rate = crops harvested / crops planted. Brier score = mean (p − y)², y = crop was present and harvest-eligible on or before day 30; "
               "reported only for modes that use the probability."]
    for cn in ["no_filter", "full", "uncertainty_fixed", "uncertainty_break_even"]:
        sub = F[F["config_name"] == cn]
        if not len(sub):
            continue
        g = _group_stats(sub)
        pm = "n/a (no probability used)"
        if cn.startswith("uncertainty") and ledger is not None:
            L = ledger[(ledger["config_name"] == cn) & ledger["experiments"].str.contains("F_feasibility_filters")]
            b = st.brier(L["predicted_completion_probability"].tolist(), L["harvest_eligible_by_horizon"].astype(int).tolist())
            results.setdefault("F_brier", {})[cn] = b
            pm = f"Brier {b['brier']:.3f} (climatology {b['climatology_brier']:.3f}, n={b['n']})" if b["n"] else "no plantings"
        results.setdefault("F", {})[cn] = g
        t10.append({"Filter mode": CONFIG_LABEL[cn], "Episodes": g["n"], "Actions evaluated": g["candidates"],
                    "Accepted actions": g["candidates"] - g["rejected"], "Rejected actions": g["rejected"],
                    "Rejection rate": _fmt(g["rejection_rate"], "rate"), "Completion rate": _fmt(g["completion_rate"], "pct"),
                    "Prediction metric": pm, "Mean filter time (ms/episode)": _fmt(g["filter_ms"], "num")})
    for cn in ["full", "uncertainty_fixed", "uncertainty_break_even"]:
        for a in agents:
            sub = F[F["agent"] == a]
            p = _paired(sub.assign(k=sub["scenario_id"]), sub["config_name"] == cn, sub["config_name"] == "no_filter", key="k")
            results.setdefault("F_pairs", {})[(a, cn)] = p
            notes10.append(f"- {AGENT_LABEL[a]}: {CONFIG_LABEL[cn]} − no filter profit = {_fmt_paired(p)} ₹ (all four profiles pooled)")
    _write_table(tables, "table_10_feasibility_performance", pd.DataFrame(t10), "Table 10 — Feasibility-filter performance (Experiment F, all agents and profiles pooled)", notes10)

    # ── Supplementary: calibration of the completion probability ───────
    if ledger is not None:
        L = ledger[ledger["experiments"].str.contains("F_feasibility_filters")]
        bins = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0001]
        rows = []
        for lo, hi in zip(bins, bins[1:]):
            m = (L["predicted_completion_probability"] >= lo) & (L["predicted_completion_probability"] < hi)
            if m.sum():
                rows.append({"Probability bin": f"[{lo:.1f}, {min(hi, 1):.1f}{']' if hi > 1 else ')'}", "Crops": int(m.sum()),
                             "Mean predicted": round(L[m]["predicted_completion_probability"].mean(), 3),
                             "Observed harvest-eligible rate": round(L[m]["harvest_eligible_by_horizon"].mean(), 3)})
        b = st.brier(L["predicted_completion_probability"].tolist(), L["harvest_eligible_by_horizon"].astype(int).tolist())
        results["calibration_all"] = b
        _write_table(tables, "table_10b_completion_probability_calibration", pd.DataFrame(rows),
                     "Table 10b — Calibration of the completion probability (all plantings in Experiment F)",
                     [f"Brier score {b['brier']:.4f} vs climatology {b['climatology_brier']:.4f} (n = {b['n']} crops; base rate {b['base_rate']:.3f}). "
                      "The probability is computed for every planting, including those made under modes that do not use it."])

    # ── Supplementary: market model ────────────────────────────────────
    M = sel("M_market_model")
    if len(M):
        rows = []
        for a in agents:
            sub = M[M["agent"] == a]
            p = _paired(sub, sub["config_name"] == "price_impact", sub["config_name"] == "full")
            results.setdefault("M", {})[a] = p
            for cn in ["full", "price_impact"]:
                g = _group_stats(sub[sub["config_name"] == cn])
                rows.append({"Agent": AGENT_LABEL[a], "Market model": "exogenous" if cn == "full" else "price_impact",
                             "Episodes": g["n"], "Mean profit (₹)": _fmt(g["mean_profit"]), "Profit SD (₹)": _fmt(g["sd_profit"]),
                             "Mean revenue (₹)": _fmt(sub[sub['config_name'] == cn]["total_revenue"].mean()),
                             "Difference vs exogenous (₹)": "reference" if cn == "full" else _fmt_paired(p)})
        _write_table(tables, "table_11_market_model", pd.DataFrame(rows), "Table 11 — Market model sensitivity (supplementary)", [])

    # ── exclusions ────────────────────────────────────────────────────
    excl = 0 if raw_summary is None else len(raw_summary) - (0 if summary is None else len(summary))
    results["excluded_episodes"] = excl
    with open(os.path.join(tables, "README.md"), "w") as f:
        f.write("# Tables\n\nAll tables are generated by `scripts/run_experiments.py` from `../data/*.csv`. "
                f"Episodes excluded from statistics (failed or incomplete): {excl}.\n\n"
                "| File | Content |\n|---|---|\n" + "\n".join(
                    f"| `{fn}` | {open(os.path.join(tables, fn)).readline().strip('# ').strip()} |"
                    for fn in sorted(os.listdir(tables)) if fn.endswith(".md") and fn != "README.md") + "\n")
    return results


# ──────────────────────────────────────────────────────────────────────────
# generated documents
# ──────────────────────────────────────────────────────────────────────────
def _p(p: Optional[Dict[str, Any]]) -> str:
    return _fmt_paired(p) + " ₹" if p else "not available"


def write_results_summary(out: str, res: Dict[str, Any], status: Dict[str, Any], chart_notes: Dict[str, str]) -> None:
    L: List[str] = ["# Results summary", "",
                    "Generated by `scripts/run_experiments.py` from `data/*.csv`; every number below is read from those files. "
                    "Brackets are 95 % confidence intervals (Student t). Differences are paired over identical scenarios. "
                    "No statistical-significance claims beyond these intervals are made.", ""]
    L.append(f"Episodes: {status.get('episodes_successful', 0)} successful, {status.get('episodes_failed', 0)} failed, "
             f"agents skipped: {', '.join(status.get('skipped_agents') or []) or 'none'}; "
             f"scenarios per profile: {status.get('n_scenarios')}. Excluded from statistics: {res.get('excluded_episodes', 0)}.")
    L.append("")
    if "A" in res:
        L.append("## A. Agent comparison (baseline, full framework)")
        for a, g in res["A"].items():
            L.append(f"- {AGENT_LABEL[a]}: mean profit ₹{g['mean_profit']:,.0f} [{g['ci_low']:,.0f}, {g['ci_high']:,.0f}], SD ₹{g['sd_profit']:,.0f}, n = {g['n']}.")
        for k, p in res.get("A_pairs", {}).items():
            x, y = k.split("-")
            L.append(f"- Paired {AGENT_LABEL[x]} − {AGENT_LABEL[y]}: {_p(p)}.")
        L.append("")
    if "B_pairs" in res:
        L.append("## B. Feasibility-filter ablation (full − no filter)")
        for (a, prof), p in res["B_pairs"].items():
            nf = res["B"][(a, prof, "no_filter")]
            L.append(f"- {AGENT_LABEL[a]}, {prof}: {_p(p)}; invalid-action rate without filter {_fmt(nf['invalid_rate'], 'rate')}.")
        L.append("")
    if "C_pairs" in res:
        L.append("## C. Decision order (FML − LFM)")
        for (a, prof), p in res["C_pairs"].items():
            L.append(f"- {AGENT_LABEL[a]}, {prof}: {_p(p)}.")
        L.append("")
    if "D" in res:
        L.append("## D. Environmental robustness (stress − baseline, same agent and seed)")
        for (a, prof), g in res["D"].items():
            if prof != "baseline":
                L.append(f"- {AGENT_LABEL[a]}, {prof}: mean profit ₹{g['mean_profit']:,.0f}; change {_p(g['diff'])}; crops destroyed/season {g['mean_destroyed']:.2f}.")
        L.append("")
    if "F" in res:
        L.append("## F. Feasibility-filter modes (all agents and profiles pooled)")
        for cn, g in res["F"].items():
            L.append(f"- {CONFIG_LABEL[cn]}: rejection rate {_fmt(g['rejection_rate'], 'rate')}, crop completion {_fmt(g['completion_rate'], 'pct')}, "
                     f"mean filter time {g['filter_ms']:.2f} ms/episode.")
        for (a, cn), p in res.get("F_pairs", {}).items():
            L.append(f"- {AGENT_LABEL[a]}: {CONFIG_LABEL[cn]} − no filter = {_p(p)}.")
        if res.get("calibration_all"):
            b = res["calibration_all"]
            L.append(f"- Completion-probability Brier score {b['brier']:.4f} vs climatology {b['climatology_brier']:.4f} "
                     f"(n = {b['n']} plantings, observed rate {b['base_rate']:.3f}, mean prediction {b['mean_prediction']:.3f}).")
        L.append("")
    if res.get("M"):
        L.append("## Market model (price_impact − exogenous)")
        for a, p in res["M"].items():
            L.append(f"- {AGENT_LABEL[a]}: {_p(p)}.")
        L.append("")
    L.append("## E. Human vs AI")
    if not res.get("E"):
        L.append("- Not run: no completed human session was recorded. No human results are reported or estimated.")
    else:
        for a, p in res["E"].items():
            L.append(f"- {AGENT_LABEL[a]} − Human: {_p(p)} over {p['n_pairs']} matched scenario(s) (positive favours the AI).")
    L += ["", "## Charts not generated", ""]
    missing = [f"- {k}: {v}" for k, v in chart_notes.items() if v.startswith("NOT GENERATED")]
    L += missing or ["- none"]
    L += ["", "See `limitations.md` before interpreting these numbers."]
    with open(os.path.join(out, "results_summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def write_readme(out: str, status: Dict[str, Any], chart_notes: Dict[str, str]) -> None:
    tables = sorted(fn for fn in os.listdir(os.path.join(out, "tables")) if fn.endswith(".csv"))
    charts = sorted(fn for fn in os.listdir(os.path.join(out, "charts")) if fn.endswith(".png"))
    exp_status = []
    for exp, desc in [("A_agent_comparison", "Main agent comparison"), ("B_feasibility_ablation", "Feasibility-filter ablation"),
                      ("C_decision_order", "Decision-order ablation"), ("D_environmental_robustness", "Environmental robustness"),
                      ("E", "Human vs AI"), ("F_feasibility_filters", "Feasibility-filter comparison"), ("M_market_model", "Market-model sensitivity")]:
        if exp == "E":
            hv = status.get("human_vs_ai", {})
            s = f"completed ({hv.get('completed_human_matches')} human match(es))" if hv.get("completed_human_matches") else "not run — no completed human session"
        else:
            s = "completed" if status.get("episodes_successful") else "not run"
        exp_status.append(f"| {exp.split('_')[0]} | {desc} | {s} |")
    text = f"""# Research outputs — Autonomous Farm AI

Everything in this folder is produced by code from executed simulations. Nothing is hand-entered.

## How to run

From the repository root, using the project's virtual environment:

```bash
# (optional) retrain PPO — about 7 minutes on a laptop CPU; writes models/ppo/farm_ppo_v2.zip + .meta.json and logs/ppo_training_*.csv
.venv/bin/python scripts/train_ppo.py --timesteps 2000000 --seed 7

# run all experiments, then build tables, charts, diagrams and summaries
.venv/bin/python scripts/run_experiments.py

# quick smoke test (5 scenarios per profile)
.venv/bin/python scripts/run_experiments.py --n 5

# rebuild tables/charts/docs from the existing CSVs only
.venv/bin/python scripts/run_experiments.py --outputs-only
```

(On Windows use `.venv\\Scripts\\python`.) Experiments never run when the app starts. Settings: `experiment_config.json`.

## Last run

- Started {status.get('started')}, finished {status.get('finished')}, runtime {status.get('runtime_s')} s
- Scenarios per profile: {status.get('n_scenarios')} (seeds 0–{(status.get('n_scenarios') or 1) - 1})
- Episodes: **{status.get('episodes_successful')} successful**, **{status.get('episodes_failed')} failed**, agents skipped: {', '.join(status.get('skipped_agents') or []) or 'none'}
- Agent status: {', '.join(f"{k}: {v.get('status')}" for k, v in status.get('agents', {}).items())}

| Exp. | Experiment | Status |
|---|---|---|
{chr(10).join(exp_status)}

## Folder layout

| Path | Content |
|---|---|
| `data/daily_episode_records.csv` | One row per simulated day per episode (actions, validity, rejections, risks, events, prices, cash, inventory, harvest, sales, losses, wealth, reward) |
| `data/episode_summary.csv` | One row per episode (all summary metrics) |
| `data/crop_ledger.csv` | One row per planted crop (planting day, predicted completion probability, outcome) |
| `data/scenario_manifest.csv`, `data/scenarios.jsonl` | Every evaluation scenario (summary + full replayable JSON) |
| `data/*_raw.csv` | Per-experiment subsets of the episode summary (`experiment_id` column) |
| `data/human_ai_comparison_raw.csv`, `data/human_ai_daily_records.csv` | Experiment E (empty until a human match is played) |
| `tables/` | Tables 1–10 (+ supplementary 10b, 11) as CSV and Markdown |
| `charts/` | Charts as 300-dpi PNG + SVG, captions in `charts/README.md` |
| `paper_assets/` | Architecture and workflow diagrams (PNG + SVG) |
| `logs/` | `run_status.json`, PPO training curves |
| `audit_report.md`, `implementation_mapping.md`, `experiment_methods.md`, `results_summary.md`, `limitations.md` | Documentation |

Tables: {', '.join(f'`{t}`' for t in tables)}

Charts: {', '.join(f'`{c}`' for c in charts)}

## Metric definitions (units)

| Metric | Definition | Unit |
|---|---|---|
| Cash | Money held | ₹ |
| Wealth | Cash + Σ inventory × current price (crops still in the field count 0) | ₹ |
| Final wealth | Wealth after the day-30 action, valued at day-30 prices | ₹ |
| Profit | Final wealth − initial wealth (₹10,000) | ₹ |
| Revenue / Costs | Σ sale proceeds / Σ planting costs, from transaction records | ₹ |
| Harvest quantity | Σ harvested produce | abstract units (not kg) |
| Crops destroyed | Crops removed by pest outbreaks | count |
| Drought set-back days | Growth days lost to droughts | crop-days |
| Completion rate | Crops harvested / crops planted (destroyed and unharvested crops count as not completed) | ratio |
| Actions attempted | Actions submitted to the environment (one per day) | count |
| Invalid-action rate | Invalid executed actions / actions attempted | ratio |
| Feasibility rejection rate | Candidates rejected by the filter / candidates evaluated by the filter | ratio |
| Reward | Δ wealth per day, −50 for an invalid action (PPO training signal) | ₹ |
| Mean, SD, 95 % CI | Sample mean; sample SD (n−1); mean ± t(0.975, n−1)·s/√n | — |
| Brier score | mean (p − y)² for completion predictions | — |
"""
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(text)


def write_mapping(out: str) -> None:
    lines = ["# Implementation mapping", "",
             "Maps each component of the research framework to the code that implements it and the test that verifies it. "
             "Generated from `src/farm_ai/research/mapping.py` (the same data as Table 2).", "",
             "Status: **Verified** = implemented and covered by the named passing test; **Partially implemented** = only part of "
             "the intended behaviour exists; **Not implemented** = absent (documented, not claimed).", ""]
    for comp, intended, actual, status_, test, notes in IMPLEMENTATION_MAPPING:
        lines += [f"## {comp} — {status_}", "", f"- Intended: {intended}", f"- Implementation: {actual}",
                  f"- Verification: `{test}`" if test != "—" else "- Verification: none", *( [f"- Notes: {notes}"] if notes else []), ""]
    lines += ["## Key files", "",
              "| File | Role |", "|---|---|",
              "| `src/farm_ai/framework/scenario.py` | Scenario profiles, seeded generation, save/load |",
              "| `src/farm_ai/environment/farm_environment.py` | Six-stage daily process, logging, crop ledger |",
              "| `src/farm_ai/framework/feasibility.py` | Feasibility filter, completion-probability DP, `decide` |",
              "| `src/farm_ai/framework/config.py` | Decision order, filter mode, market model |",
              "| `src/farm_ai/models/market.py` | Exogenous and price-impact markets |",
              "| `src/farm_ai/framework/agents.py` | Random / rule-based / PPO policies, checkpoint check |",
              "| `src/farm_ai/framework/episode.py` | Episode runner and summary metrics |",
              "| `src/farm_ai/api/services/match_service.py` | Human-vs-AI matches |",
              "| `src/farm_ai/research/` | Experiment runner, statistics, tables, charts, diagrams |",
              "| `frontend/src/pages/VersusPage.tsx` | Human-vs-AI user interface |", ""]
    with open(os.path.join(out, "implementation_mapping.md"), "w") as f:
        f.write("\n".join(lines))


def build_all(out: str, config_path: str, root: str) -> None:
    from .charts import build_charts
    from .diagrams import build_diagrams

    with open(config_path) as f:
        cfg = json.load(f)
    status_path = os.path.join(out, "logs", "run_status.json")
    status = {}
    if os.path.isfile(status_path):
        with open(status_path) as f:
            status = json.load(f)
    res = build_tables(out, cfg, status)
    notes = build_charts(out)
    build_diagrams(out)
    write_results_summary(out, res, status, notes)
    write_readme(out, status, notes)
    write_mapping(out)
