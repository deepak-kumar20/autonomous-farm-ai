"""Matplotlib charts (300-dpi PNG + SVG) generated only from saved CSV data.

Conventions: one y-axis per panel (no dual axes); agent colours fixed per entity; legends present
whenever a panel has ≥ 2 series; values in ink colours; 95 % CI error bars = mean ± t·s/√n.
A chart whose inputs are missing is replaced by a status note in charts/README.md.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats as st  # noqa: E402
from .outputs import AGENT_COLOR, AGENT_LABEL, CONFIG_LABEL, FILTER_COLOR, GRID, INK, INK2, PROFILE_ORDER, _ok, _read  # noqa: E402
from ..config.settings import STARTING_CASH, CROP_CONFIG  # noqa: E402

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 100,
})


def _save(fig, charts_dir: str, name: str) -> str:
    if fig.legends:
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(charts_dir, f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)
    return name


def _ci_half(xs) -> float:
    c = st.ci95(st.clean(list(xs)))
    return 0.0 if c is None else (c[1] - c[0]) / 2


def _money_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"₹{v:,.0f}"))


def _bars(ax, groups: List[str], series: List[str], values: Dict, errors: Dict, colors: Dict, labels: Dict):
    w = 0.8 / max(1, len(series))
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        vals = [values.get((g, s), np.nan) for g in groups]
        errs = [errors.get((g, s), 0) for g in groups]
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w * 0.92, yerr=errs, capsize=2.5, color=colors[s],
               label=labels[s], error_kw={"elinewidth": 0.9, "ecolor": INK2})
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.axhline(0, color=INK2, linewidth=0.8)


def build_charts(out: str) -> Dict[str, str]:
    data, charts = os.path.join(out, "data"), os.path.join(out, "charts")
    os.makedirs(charts, exist_ok=True)
    notes: Dict[str, str] = {}
    S = _ok(_read(os.path.join(data, "episode_summary.csv")))
    daily = _read(os.path.join(data, "daily_episode_records.csv"))

    def sel(exp):
        return S[S["experiments"].str.split(";").apply(lambda xs: exp in xs)] if S is not None else None

    agents = [a for a in ["random", "rule_based", "ppo"] if S is not None and a in set(S["agent"])]

    # 1 — wealth trajectory
    hv_daily = _read(os.path.join(data, "human_ai_daily_records.csv"))
    if daily is not None and S is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        A = daily[daily["experiments"].str.contains("A_agent_comparison")]
        for a in agents:
            d = A[A["agent"] == a]
            g = d.groupby("day")["wealth_end"]
            m, h = g.mean(), g.apply(_ci_half)
            days = np.concatenate([[0], m.index.values])
            mm = np.concatenate([[STARTING_CASH], m.values])
            hh = np.concatenate([[0], h.values])
            ax.plot(days, mm, color=AGENT_COLOR[a], linewidth=2, label=f"{AGENT_LABEL[a]} (mean of {d['run_id'].nunique()})")
            ax.fill_between(days, mm - hh, mm + hh, color=AGENT_COLOR[a], alpha=0.15, linewidth=0)
        human_note = ""
        if hv_daily is not None and "human" in set(hv_daily["player_type"]):
            hd = hv_daily[hv_daily["player_type"] == "human"]
            g = hd.groupby("day")["wealth_end"].mean()
            ax.plot(np.concatenate([[0], g.index.values]), np.concatenate([[STARTING_CASH], g.values]), color=AGENT_COLOR["human"],
                    linewidth=2, linestyle="--", label=f"Human (mean of {hd['match_id'].nunique()} match(es))")
        else:
            human_note = " No completed human session exists yet, so the human curve is absent."
        ax.axhline(STARTING_CASH, color=INK2, linewidth=0.8, linestyle=":")
        ax.set(title="Net wealth over the 30-day season (baseline scenarios)", xlabel="Day (end of day)", ylabel="Wealth (₹)", xlim=(0, 30))
        _money_axis(ax)
        ax.legend(loc="upper left")
        notes["chart_1_wealth_over_30_days"] = _save(fig, charts, "chart_1_wealth_over_30_days") and \
            "Mean end-of-day wealth (cash + inventory at current price) across baseline scenarios, full framework; bands = 95 % CI of the mean. " \
            "Growing crops carry no value until harvested, which produces the early dip." + human_note

    # 2 — agent profit comparison
    if S is not None:
        A = sel("A_agent_comparison")
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        vals = [A[A["agent"] == a]["profit"].mean() for a in agents]
        errs = [_ci_half(A[A["agent"] == a]["profit"]) for a in agents]
        bars = ax.bar([AGENT_LABEL[a] for a in agents], vals, yerr=errs, capsize=4, color=[AGENT_COLOR[a] for a in agents],
                      error_kw={"elinewidth": 1, "ecolor": INK2})
        for b, v, e in zip(bars, vals, errs):
            ax.annotate(f"₹{v:,.0f}", (b.get_x() + b.get_width() / 2, v + e), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=8)
        ax.set(title="Mean season profit by agent (Experiment A)", ylabel="Profit (₹)")
        _money_axis(ax)
        notes["chart_2_agent_profit_comparison"] = _save(fig, charts, "chart_2_agent_profit_comparison") and \
            f"Mean profit over {A['scenario_id'].nunique()} baseline scenarios (full framework); error bars = 95 % CI."

    # 3 — robustness
    if S is not None:
        D = sel("D_environmental_robustness")
        profs = [p for p in PROFILE_ORDER if p in set(D["scenario_profile"])]
        vals = {(p, a): D[(D["scenario_profile"] == p) & (D["agent"] == a)]["profit"].mean() for p in profs for a in agents}
        errs = {(p, a): _ci_half(D[(D["scenario_profile"] == p) & (D["agent"] == a)]["profit"]) for p in profs for a in agents}
        fig, ax = plt.subplots(figsize=(7, 4))
        _bars(ax, profs, agents, vals, errs, AGENT_COLOR, AGENT_LABEL)
        ax.set(title="Profit under environmental stress (Experiment D)", ylabel="Mean profit (₹)", xlabel="Scenario profile")
        _money_axis(ax)
        ax.legend()
        notes["chart_3_weather_pest_robustness"] = _save(fig, charts, "chart_3_weather_pest_robustness") and \
            "Mean profit per agent and scenario profile (full framework), error bars = 95 % CI. dry: drought 15 %/day; pest: outbreaks 20 %/day; combined: both."

    # 4 — feasibility ablation
    if S is not None:
        B = sel("B_feasibility_ablation")
        profs = [p for p in PROFILE_ORDER if p in set(B["scenario_profile"])]
        fig, axes = plt.subplots(1, len(profs), figsize=(3.6 * len(profs), 3.8), sharey=True, squeeze=False)
        for ax, p in zip(axes[0], profs):
            sub = B[B["scenario_profile"] == p]
            groups = [AGENT_LABEL[a] for a in agents]
            vals = {(AGENT_LABEL[a], c): sub[(sub["agent"] == a) & (sub["config_name"] == c)]["profit"].mean() for a in agents for c in ["no_filter", "full"]}
            errs = {(AGENT_LABEL[a], c): _ci_half(sub[(sub["agent"] == a) & (sub["config_name"] == c)]["profit"]) for a in agents for c in ["no_filter", "full"]}
            _bars(ax, groups, ["no_filter", "full"], vals, errs, {"no_filter": "#2a78d6", "full": "#eb6834"},
                  {"no_filter": "No filter", "full": "Deterministic filter (full framework)"})
            ax.set_title(f"{p} scenarios")
            _money_axis(ax)
        axes[0][0].set_ylabel("Mean profit (₹)")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.suptitle("Feasibility-filter ablation (Experiment B)", fontsize=11)
        fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        notes["chart_4_feasibility_ablation"] = _save(fig, charts, "chart_4_feasibility_ablation") and \
            "Same agent, same scenarios; only the pre-execution filter changes. Error bars = 95 % CI."

    # 5 — decision order
    if S is not None:
        C = sel("C_decision_order")
        profs = [p for p in PROFILE_ORDER if p in set(C["scenario_profile"])]
        fig, axes = plt.subplots(1, len(profs), figsize=(3.6 * len(profs), 3.8), sharey=True, squeeze=False)
        for ax, p in zip(axes[0], profs):
            sub = C[C["scenario_profile"] == p]
            groups = [AGENT_LABEL[a] for a in agents]
            vals = {(AGENT_LABEL[a], c): sub[(sub["agent"] == a) & (sub["config_name"] == c)]["profit"].mean() for a in agents for c in ["full", "order_fml"]}
            errs = {(AGENT_LABEL[a], c): _ci_half(sub[(sub["agent"] == a) & (sub["config_name"] == c)]["profit"]) for a in agents for c in ["full", "order_fml"]}
            _bars(ax, groups, ["full", "order_fml"], vals, errs, {"full": "#eb6834", "order_fml": "#6250d6"},
                  {"full": "LFM (losses before decision)", "order_fml": "FML (losses after decision)"})
            ax.set_title(f"{p} scenarios")
            _money_axis(ax)
        axes[0][0].set_ylabel("Mean profit (₹)")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.suptitle("Decision-order ablation (Experiment C)", fontsize=11)
        fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        notes["chart_5_decision_order_ablation"] = _save(fig, charts, "chart_5_decision_order_ablation") and \
            "Only the position of the loss stage changes (deterministic filter, exogenous market). Error bars = 95 % CI."

    # 6 — human vs AI
    hv = _read(os.path.join(data, "human_ai_comparison_raw.csv"))
    if hv is not None and "human" in set(hv["player_type"]):
        hv = hv.assign(who=np.where(hv["player_type"] == "human", "human", hv["player"]))
        order = ["human"] + [a for a in agents if a in set(hv["who"])]
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        vals = [hv[hv["who"] == w]["profit"].mean() for w in order]
        errs = [_ci_half(hv[hv["who"] == w]["profit"]) for w in order]
        ax.bar([AGENT_LABEL[w] for w in order], vals, yerr=errs, capsize=4, color=[AGENT_COLOR[w] for w in order],
               error_kw={"elinewidth": 1, "ecolor": INK2})
        nm = hv["match_id"].nunique()
        ax.set(title=f"Human vs AI on the same scenario{'s' if nm > 1 else ''} (n = {nm})", ylabel="Profit (₹)")
        _money_axis(ax)
        notes["chart_6_human_vs_ai"] = _save(fig, charts, "chart_6_human_vs_ai") and \
            ("Single matched scenario: a case study, not a general comparison." if nm == 1 else "Mean profit over matched scenarios; error bars = 95 % CI.")
    else:
        notes["chart_6_human_vs_ai"] = "NOT GENERATED — no completed human session exists. Play a match on the Versus page, then rerun scripts/run_experiments.py."

    # 7 — PPO training curve
    tr = _read(os.path.join(out, "logs", "ppo_training_eval.csv"))
    if tr is not None:
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
        for fm, col, lab in [("none", "#2a78d6", "Greedy policy, no filter"), ("deterministic", "#eb6834", "Greedy policy + deterministic filter")]:
            d = tr[tr["filter_mode"] == fm].drop_duplicates("timesteps", keep="last")
            a1.plot(d["timesteps"], d["mean_profit"], color=col, linewidth=2, label=lab, marker="o", markersize=3)
            if fm == "none":
                a2.plot(d["timesteps"], d["invalid_rate"], color=col, linewidth=2, marker="o", markersize=3)
        a1.set(title="PPO validation during training (20 held-out scenarios)", ylabel="Mean profit (₹)")
        _money_axis(a1)
        a1.legend(loc="lower right")
        a2.set(ylabel="Invalid-action rate (no filter)", xlabel="Training timesteps")
        a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
        notes["chart_7_ppo_training_curve"] = _save(fig, charts, "chart_7_ppo_training_curve") and \
            "Validation scenarios: seeds 900000–900019, profiles cycled. Profit is unscaled terminal profit (₹), not training reward."
    else:
        notes["chart_7_ppo_training_curve"] = "NOT GENERATED — research_outputs/logs/ppo_training_eval.csv missing (run scripts/train_ppo.py)."

    # 8 — market prices and sales
    if daily is not None:
        sid = "S-baseline-0"
        d = daily[(daily["scenario_id"] == sid) & (daily["config_name"] == "full")]
        if len(d):
            crops = list(CROP_CONFIG)
            fig, axes = plt.subplots(1 + len(agents), 1, figsize=(7, 2.0 * (1 + len(agents))), sharex=True)
            ref = d[d["agent"] == agents[0]]
            crop_col = {"Wheat": "#2a78d6", "Rice": "#eb6834", "Corn": "#1baf7a"}
            for c in crops:
                axes[0].plot(ref["day"], ref[f"price_{c.lower()}"], color=crop_col[c], linewidth=2, label=c)
            axes[0].set(ylabel="Price (₹/unit)", title=f"Market prices and units sold — scenario {sid}")
            axes[0].legend(ncol=3, loc="upper left")
            for ax, a in zip(axes[1:], agents):
                da = d[d["agent"] == a]
                bottom = np.zeros(30)
                for c in crops:
                    q = np.array([da[(da["day"] == t) & (da["sold_crop"] == c)]["sold_qty"].sum() for t in range(1, 31)])
                    ax.bar(range(1, 31), q, bottom=bottom, color=crop_col[c], width=0.8, label=c)
                    bottom += q
                ax.set(ylabel=f"{AGENT_LABEL[a]}: units sold")
            axes[-1].set_xlabel("Day")
            notes["chart_8_market_price_and_sales"] = _save(fig, charts, "chart_8_market_price_and_sales") and \
                "Exogenous market: the top panel is identical for every agent; lower panels show each agent's daily sales (units) in the same scenario."

    # 9 — crop losses by scenario
    if S is not None:
        D = sel("D_environmental_robustness")
        profs = [p for p in PROFILE_ORDER if p in set(D["scenario_profile"])]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6))
        for ax, col, ttl, yl in [(a1, "crops_destroyed", "Crops destroyed by pests", "Crops per season"),
                                 (a2, "drought_setback_days", "Growth days lost to drought", "Crop-days per season")]:
            vals = {(p, a): D[(D["scenario_profile"] == p) & (D["agent"] == a)][col].mean() for p in profs for a in agents}
            errs = {(p, a): _ci_half(D[(D["scenario_profile"] == p) & (D["agent"] == a)][col]) for p in profs for a in agents}
            _bars(ax, profs, agents, vals, errs, AGENT_COLOR, AGENT_LABEL)
            ax.set(title=ttl, ylabel=yl)
        a1.legend()
        notes["chart_9_crop_losses_by_scenario"] = _save(fig, charts, "chart_9_crop_losses_by_scenario") and \
            "Mean per episode with 95 % CI; losses depend on both the scenario and how many crops the agent exposes."

    # 10 — feasibility decisions
    if daily is not None:
        F = daily[daily["experiments"].str.contains("F_feasibility_filters")]
        modes = [m for m in ["full", "uncertainty_fixed", "uncertainty_break_even"] if m in set(F["config_name"])]
        if modes:
            groups = {
                "Structural (field, maturity, cash, stock)": {"FIELD_EMPTY", "FIELD_OCCUPIED", "NOT_MATURE", "NO_INVENTORY", "INSUFFICIENT_CASH", "BAD_TARGET"},
                "Horizon (cannot finish by day 30)": {"HORIZON"},
                "Low completion probability": {"LOW_COMPLETION_PROBABILITY"},
            }
            gcol = ["#2a78d6", "#eb6834", "#1baf7a"]
            fig, ax = plt.subplots(figsize=(9, 4))
            bottom = np.zeros(len(modes))
            codes_per_mode = [F[F["config_name"] == m]["rejection_codes"].dropna().astype(str).str.split(";").explode() for m in modes]
            for i, (gname, members) in enumerate(groups.items()):
                vals = np.array([int(c.isin(members).sum()) for c in codes_per_mode])
                ax.bar([CONFIG_LABEL[m].replace(" (", "\n(") for m in modes], vals, bottom=bottom, color=gcol[i], label=gname, width=0.6)
                bottom += vals
            ax.set(title="Rejected candidate actions by reason (Experiment F)", ylabel="Rejected candidates (count)")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.legend(title="Rejection reason", loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
            acc = [int((F[F["config_name"] == m]["candidates_evaluated"] - F[F["config_name"] == m]["feasibility_rejections"]).sum()) for m in modes]
            for x, (b, a_) in enumerate(zip(bottom, acc)):
                ax.annotate(f"rejected {int(b):,} · accepted {a_:,}", (x, b), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
            notes["chart_10_feasibility_decisions"] = _save(fig, charts, "chart_10_feasibility_decisions") and \
                "Counts over all agents and profiles. The no-filter arm evaluates nothing and is omitted."

    # 11 — calibration (supplementary)
    led = _read(os.path.join(data, "crop_ledger.csv"))
    if led is not None:
        L = led[led["experiments"].str.contains("F_feasibility_filters")]
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(L["predicted_completion_probability"], bins) - 1, 0, 9)
        g = L.assign(b=idx).groupby("b").agg(p=("predicted_completion_probability", "mean"), y=("harvest_eligible_by_horizon", "mean"),
                                             n=("crop_uid", "size"))
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=1, label="Perfect calibration")
        ax.plot(g["p"], g["y"], color="#2a78d6", marker="o", linewidth=2, label="Observed")
        for _, r in g.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r["p"], r["y"]), xytext=(4, -10), textcoords="offset points", fontsize=7, color=INK2)
        b = st.brier(L["predicted_completion_probability"].tolist(), L["harvest_eligible_by_horizon"].astype(int).tolist())
        ax.set(title=f"Completion-probability calibration\nBrier {b['brier']:.3f} (climatology {b['climatology_brier']:.3f})",
               xlabel="Predicted P(harvest-eligible by day 30)", ylabel="Observed rate", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(loc="upper left")
        notes["chart_11_completion_probability_calibration"] = _save(fig, charts, "chart_11_completion_probability_calibration") and \
            "Reliability diagram over every planting in Experiment F (ten equal-width bins)."

    for name in ["chart_1_wealth_over_30_days", "chart_2_agent_profit_comparison", "chart_3_weather_pest_robustness",
                 "chart_4_feasibility_ablation", "chart_5_decision_order_ablation", "chart_8_market_price_and_sales",
                 "chart_9_crop_losses_by_scenario", "chart_10_feasibility_decisions"]:
        notes.setdefault(name, "NOT GENERATED — required experiment data missing (run scripts/run_experiments.py).")
    with open(os.path.join(charts, "README.md"), "w") as f:
        f.write("# Charts\n\nGenerated by `scripts/run_experiments.py` from `../data/` and `../logs/`. Each chart is saved as 300-dpi PNG and SVG.\n\n")
        for k in sorted(notes, key=lambda s: int(s.split("_")[1])):
            f.write(f"## {k}\n\n" + (f"![{k}]({k}.png)\n\n" if not notes[k].startswith("NOT GENERATED") else "") + f"{notes[k]}\n\n")
    return notes
