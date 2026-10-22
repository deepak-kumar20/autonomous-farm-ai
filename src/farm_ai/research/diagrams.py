"""Publication diagrams drawn with matplotlib (PNG 300 dpi + SVG)."""
from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2 = "#0b0b0b", "#52514e"
FILL = {"data": "#e8f1fb", "engine": "#fdeee6", "agent": "#e6f6ef", "ui": "#fdf3dc", "out": "#efedfb"}
EDGE = {"data": "#2a78d6", "engine": "#eb6834", "agent": "#1baf7a", "ui": "#c98500", "out": "#6250d6"}


def _box(ax, x, y, w, h, text, kind, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", fc=FILL[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True)


def _arrow(ax, p0, p1, text=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=INK2, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.015, text, ha="center", va="bottom", fontsize=7, color=INK2)


def _fig(w=9, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _save(fig, d, name):
    fig.tight_layout()
    fig.savefig(os.path.join(d, f"{name}.png"), dpi=300)
    fig.savefig(os.path.join(d, f"{name}.svg"))
    plt.close(fig)


def build_diagrams(out: str) -> List[str]:
    d = os.path.join(out, "paper_assets")
    os.makedirs(d, exist_ok=True)

    # 1 — system architecture
    fig, ax = _fig(9.5, 5.4)
    ax.text(0.5, 0.97, "System architecture — Autonomous Farm AI", ha="center", va="top", fontsize=12, color=INK, weight="bold")
    _box(ax, 0.03, 0.70, 0.22, 0.16, "React frontend\nFarm · Market · Barn · Weather\nAI Farmer · Versus · Journal", "ui")
    _box(ax, 0.32, 0.70, 0.30, 0.16, "FastAPI\n/api/farm · /api/ppo · /api/scenarios\n/api/matches · /api/agents", "ui")
    _box(ax, 0.70, 0.70, 0.27, 0.16, "Match store (JSON)\ndata/scenarios · data/matches\nhuman + AI episodes", "data")
    _box(ax, 0.03, 0.40, 0.22, 0.18, "Scenario generator\nseeded per-subsystem streams\nrisks + events + prices", "data")
    _box(ax, 0.32, 0.40, 0.30, 0.18, "FarmEnvironment (Gymnasium)\nsix-stage daily process\ntransition log + crop ledger", "engine")
    _box(ax, 0.70, 0.40, 0.27, 0.18, "Feasibility filter\nnone · deterministic · uncertainty\n(completion-probability DP)", "engine")
    _box(ax, 0.03, 0.10, 0.22, 0.18, "Policies\nRandom · Rule-based\nPPO (SB3, checkpoint check)", "agent")
    _box(ax, 0.32, 0.10, 0.30, 0.18, "Experiment runner\nruns each (agent, scenario, config) once\nCSV datasets", "out")
    _box(ax, 0.70, 0.10, 0.27, 0.18, "Outputs\ntables · charts · diagrams\nresults summary", "out")
    _arrow(ax, (0.25, 0.78), (0.32, 0.78), "HTTP/JSON")
    _arrow(ax, (0.62, 0.78), (0.70, 0.78), "save/load")
    _arrow(ax, (0.47, 0.70), (0.47, 0.58), "actions")
    _arrow(ax, (0.25, 0.49), (0.32, 0.49), "scenario")
    _arrow(ax, (0.70, 0.49), (0.62, 0.49), "accept/reject")
    _arrow(ax, (0.14, 0.28), (0.36, 0.40), "ranked candidates", rad=-0.15)
    _arrow(ax, (0.47, 0.40), (0.47, 0.28), "episodes")
    _arrow(ax, (0.62, 0.19), (0.70, 0.19), "CSV")
    _save(fig, d, "system_architecture")

    # 2 — six-stage workflow
    fig, ax = _fig(10, 4.6)
    ax.text(0.5, 0.97, "Six-stage daily decision process (default order: Loss → Field → Market)", ha="center", va="top", fontsize=12, color=INK, weight="bold")
    stages: List[Tuple[str, str]] = [
        ("1. Resolve\nlosses", "scenario events\nfor day t:\ndrought set-back,\npest outbreak"),
        ("2. Candidate\nactions", "human request\nor policy ranking\nover 24 actions"),
        ("3. Feasibility\nfilter", "cash · field ·\nmaturity · stock ·\nhorizon ·\nP(complete)"),
        ("4. Field\naction", "PLANT or\nHARVEST\n(cash, field,\ncrop ledger)"),
        ("5. Market\naction", "SELL at the\ncurrent price\n(no negative\nstock)"),
        ("6. Update\nand log", "growth, market\nclock, reward,\nrecord, day t+1"),
    ]
    w, gap = 0.142, 0.022
    for i, (t, s_) in enumerate(stages):
        x = 0.015 + i * (w + gap)
        kind = "data" if i == 0 else ("agent" if i == 1 else ("engine" if i in (2, 3, 4) else "out"))
        _box(ax, x, 0.36, w, 0.46, "", kind)
        ax.text(x + w / 2, 0.76, t, ha="center", va="top", fontsize=9, color=INK, weight="bold")
        ax.text(x + w / 2, 0.58, s_, ha="center", va="center", fontsize=7.6, color=INK)
        if i < 5:
            _arrow(ax, (x + w, 0.59), (x + w + gap, 0.59))
    x2 = 0.015 + 2 * (w + gap) + w / 2
    x1 = 0.015 + 1 * (w + gap) + w / 2
    _arrow(ax, (x2, 0.36), (x1, 0.36), rad=-0.6)
    ax.text((x1 + x2) / 2, 0.22, "rejected → next candidate\n(nothing changes)", ha="center", va="center", fontsize=7.5, color=INK2)
    ax.text(0.5, 0.07, "Exactly one action is taken per day, so stages 4 and 5 are mutually exclusive. The FML ablation moves stage 1 to the end of\n"
            "the day, after crop growth (the original simulator's order).", ha="center", va="center", fontsize=8, color=INK2)
    _save(fig, d, "six_stage_workflow")

    # 3 — human vs AI
    fig, ax = _fig(9.5, 4.4)
    ax.text(0.5, 0.96, "Human-vs-AI evaluation on one saved scenario", ha="center", va="top", fontsize=12, color=INK, weight="bold")
    _box(ax, 0.02, 0.40, 0.17, 0.26, "A. Select scenario\nprofile + seed\n→ saved JSON\n(risks shown, events hidden)", "data")
    _box(ax, 0.25, 0.58, 0.22, 0.22, "B. Human plays 30 days\nsame env + filter\nrejections explained", "ui")
    _box(ax, 0.25, 0.18, 0.22, 0.22, "C. AI replays\nfresh env from saved scenario\nRandom / Rule / PPO", "agent")
    _box(ax, 0.53, 0.58, 0.18, 0.22, "Human episode\nrecords · summary · ledger", "out")
    _box(ax, 0.53, 0.18, 0.18, 0.22, "AI episode(s)\nrecords · summary · ledger", "out")
    _box(ax, 0.77, 0.36, 0.21, 0.28, "D. Compare\nprofit · wealth path\nharvest · losses\nrejections · actions", "engine")
    _arrow(ax, (0.19, 0.58), (0.25, 0.68))
    _arrow(ax, (0.19, 0.48), (0.25, 0.30))
    _arrow(ax, (0.47, 0.69), (0.53, 0.69))
    _arrow(ax, (0.47, 0.29), (0.53, 0.29))
    _arrow(ax, (0.71, 0.69), (0.77, 0.56))
    _arrow(ax, (0.71, 0.29), (0.77, 0.44))
    ax.text(0.5, 0.06, "Identical: exogenous weather, pest and price-shock draws; starting state; rules; filter mode. "
            "Player-dependent: actions, and prices only under the price-impact market.", ha="center", fontsize=8, color=INK2)
    _save(fig, d, "human_vs_ai_workflow")

    # 4 — scenario generation
    fig, ax = _fig(9.5, 4.4)
    ax.text(0.5, 0.96, "Environmental scenario generation", ha="center", va="top", fontsize=12, color=INK, weight="bold")
    _box(ax, 0.02, 0.40, 0.15, 0.22, "Profile\n(baseline, dry,\npest, combined)\n+ seed", "data")
    _box(ax, 0.22, 0.40, 0.16, 0.22, "SeedSequence(seed)\n→ 9 independent\nstreams", "engine")
    rows = [("Risk paths  p_t = base·exp(σz_t − σ²/2)\nz_t: AR(1), ρ = 0.8", 0.70),
            ("Events  e_t = 1[U_t < p_t]\n(drought, pest, price shock)", 0.44),
            ("Prices  random walk ±5 %, clamp\n[0.5, 1.5]×base, × shock factor", 0.18)]
    for txt, y in rows:
        _box(ax, 0.44, y, 0.28, 0.17, txt, "engine", fs=7.8)
        _arrow(ax, (0.38, 0.51), (0.44, y + 0.085))
    _box(ax, 0.78, 0.40, 0.2, 0.22, "Scenario JSON\nid S-<profile>-<seed>\nreplayable", "out")
    for _, y in rows:
        _arrow(ax, (0.72, y + 0.085), (0.78, 0.51))
    _arrow(ax, (0.17, 0.51), (0.22, 0.51))
    ax.text(0.5, 0.05, "Changing one profile rate re-uses the other streams' draws (common random numbers across scenario groups).",
            ha="center", fontsize=8, color=INK2)
    _save(fig, d, "environmental_scenario_workflow")
    return ["system_architecture", "six_stage_workflow", "human_vs_ai_workflow", "environmental_scenario_workflow"]
