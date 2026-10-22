"""Small, dependency-free statistics used by the table generators.

* Sample standard deviation:  s = sqrt( Σ (x_i − x̄)² / (n − 1) )  — undefined (None) for n < 2.
* 95 % confidence interval for a mean:  x̄ ± t_{0.975, n−1} · s / √n  (Student t).
  Critical values are tabulated for df = 1…30 and interpolated linearly in 1/df between
  30, 40, 60, 120 and ∞ (max interpolation error < 0.002 for df > 30).
* Paired difference:  d_i = x_i(A) − x_i(B) over matched scenarios; the CI of the mean of d uses the
  same formula with s computed from the d_i.
* Brier score: mean of (p_i − y_i)²; reference "climatology" score: ȳ(1 − ȳ).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_T975_TAIL = [(30, 2.042), (40, 2.021), (60, 2.000), (120, 1.980), (math.inf, 1.960)]


def t_crit_975(df: int) -> float:
    if df < 1:
        raise ValueError("df must be >= 1")
    if df in _T975:
        return _T975[df]
    for (d0, t0), (d1, t1) in zip(_T975_TAIL, _T975_TAIL[1:]):
        if d0 <= df <= d1:
            x0, x1, x = 1 / d0, (0.0 if d1 == math.inf else 1 / d1), 1 / df
            return t0 + (t1 - t0) * (x0 - x) / (x0 - x1)
    return 1.960


def clean(xs: Iterable[Optional[float]]) -> List[float]:
    return [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def sd(xs: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def ci95(xs: Sequence[float]) -> Optional[tuple]:
    n = len(xs)
    s = sd(xs)
    if s is None:
        return None
    h = t_crit_975(n - 1) * s / math.sqrt(n)
    m = sum(xs) / n
    return (m - h, m + h)


def describe(xs: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    v = clean(xs)
    c = ci95(v)
    return {"n": len(v), "mean": mean(v), "sd": sd(v), "ci_low": c[0] if c else None, "ci_high": c[1] if c else None}


def paired(a: Dict, b: Dict) -> Dict[str, Optional[float]]:
    """Paired differences a − b over keys present in both dicts."""
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    out = describe(d)
    out["n_pairs"] = len(keys)
    out["share_positive"] = (sum(1 for x in d if x > 0) / len(d)) if d else None
    return out


def brier(p: Sequence[float], y: Sequence[int]) -> Dict[str, Optional[float]]:
    n = len(p)
    if n == 0:
        return {"n": 0, "brier": None, "base_rate": None, "climatology_brier": None, "mean_prediction": None}
    b = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n
    yb = sum(y) / n
    return {"n": n, "brier": b, "base_rate": yb, "climatology_brier": yb * (1 - yb), "mean_prediction": sum(p) / n}


def fmt_money(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:,.0f}"


def fmt_ci(lo: Optional[float], hi: Optional[float]) -> str:
    return "n/a (n<2)" if lo is None else f"[{lo:,.0f}, {hi:,.0f}]"
