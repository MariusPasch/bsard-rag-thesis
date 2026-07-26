"""
Derive data-driven AQS weights from the 19-system RQ3 evaluation.

The original AQS used arbitrary weights (umbrela=0.4375, erag=0.3750,
ragas_wa=0.1875 — ratio 7:6:3). This script discards those and fits new
weights that maximise system-level Spearman rank correlation between the
composite AQS and gold Recall@10.

Method
------
- Inputs: 19 systems' (UMBRELA-mean, eRAG-mean, RAGAS-WA-mean, Recall@10,
  NDCG@10, MAP@10) loaded from output/results/RQ3/*_test.json.
- Constraints: w_i >= 0, sum(w_i) == 1.
- Search: simplex grid (step 0.05) + SLSQP refinement starting from grid
  maximum (Spearman is non-smooth; SLSQP handles the convex constraint
  set, the grid avoids local optima).
- LOO: leave-one-system-out; fit on 18, then for each fold i compute
  Spearman(AQS_w_i across all 19 systems, gold across all 19). The mean
  across the 19 folds is "rho_loo_mean"; per-weight 95% CIs come from the
  empirical distribution of the 19 fitted weight vectors.
- Multi-metric variant: same procedure with objective = mean Spearman
  against [Recall@10, NDCG@10, MAP@10].
- Sanity check: Spearman of each component alone vs Recall@10, plus the
  old 7:6:3 weights.

Output
------
output/results/RQ3/aqs_weights.json — see schema in the prompt.

Also rewrites output/results/RQ3/summary.csv to add a `T3/AQS_dd` column
(data-driven full-data AQS) without touching the existing `T3/AQS`.
"""

from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT      = Path(__file__).parent.parent.parent
_RQ3_DIR   = _ROOT / "output" / "results" / "RQ3"
_SUMMARY   = _RQ3_DIR / "summary.csv"
_OUT_JSON  = _RQ3_DIR / "aqs_weights.json"

COMPONENTS = ["umbrela", "erag", "ragas_wa"]
COMPONENT_COLS = {
    "umbrela":  "T3/umbrela/mean",
    "erag":     "T3/erag/mean",
    "ragas_wa": "T3/ragas_wa/mean",
}
GOLD_PRIMARY = "Recall@10"
GOLD_MULTI   = ["Recall@10", "NDCG@10", "MAP@10"]
OLD_WEIGHTS  = {"umbrela": 0.4375, "erag": 0.3750, "ragas_wa": 0.1875}  # 7:6:3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_system_table() -> pd.DataFrame:
    """Load 19-system table with components + gold metrics from RQ3 records."""
    rows = []
    for f in sorted(_RQ3_DIR.glob("*_test.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        m = d["subset_metrics"]["metrics"]
        rows.append({
            "experiment_id":  d["experiment_id"],
            "family":         d["family"],
            "umbrela":        m.get("T3/umbrela/mean"),
            "erag":           m.get("T3/erag/mean"),
            "ragas_wa":       m.get("T3/ragas_wa/mean"),
            "Recall@10":      m.get("Recall@10"),
            "NDCG@10":        m.get("NDCG@10"),
            "MAP@10":         m.get("MAP@10"),
        })
    df = pd.DataFrame(rows).dropna(
        subset=[*COMPONENTS, *GOLD_MULTI]
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Weight fitting
# ---------------------------------------------------------------------------

def aqs(weights: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Compute AQS = sum_i w_i * components[:, i]. components shape (n_sys, 3)."""
    return components @ weights


def neg_spearman_single(weights: np.ndarray, components: np.ndarray, gold: np.ndarray) -> float:
    pred = aqs(weights, components)
    if pred.std() == 0:
        return 1.0
    rho = spearmanr(pred, gold).statistic
    return -float(rho) if not np.isnan(rho) else 1.0


def neg_spearman_multi(weights: np.ndarray, components: np.ndarray, golds: np.ndarray) -> float:
    """golds shape (n_sys, k_metrics). Negative mean Spearman across metrics."""
    pred = aqs(weights, components)
    if pred.std() == 0:
        return 1.0
    rhos = [spearmanr(pred, golds[:, j]).statistic for j in range(golds.shape[1])]
    rhos = [r for r in rhos if not np.isnan(r)]
    if not rhos:
        return 1.0
    return -float(np.mean(rhos))


def simplex_grid(step: float) -> list[tuple[float, float, float]]:
    """All (w1, w2, w3) on the simplex with grid spacing `step`."""
    n = int(round(1.0 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            pts.append((i * step, j * step, k * step))
    return pts


def fit_weights(components: np.ndarray, gold: np.ndarray, multi: bool = False) -> tuple[np.ndarray, float]:
    """Return (best_weights, best_neg_objective) — neg-Spearman or neg-mean-Spearman."""
    obj = (lambda w: neg_spearman_multi(w, components, gold)) if multi \
        else (lambda w: neg_spearman_single(w, components, gold))

    # Coarse simplex grid
    grid = simplex_grid(0.05)
    grid_vals = [(obj(np.array(w)), w) for w in grid]
    grid_vals.sort(key=lambda t: t[0])  # ascending = best first (neg-Spearman)
    best_neg, best_w = grid_vals[0]
    best_w = np.array(best_w)

    # SLSQP refinement
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bounds = [(0.0, 1.0)] * 3
    try:
        res = minimize(obj, x0=best_w, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-8, "maxiter": 200})
        if res.success and res.fun <= best_neg + 1e-9:
            # Spearman is rank-based and step-flat so SLSQP often plateaus;
            # prefer SLSQP result only if strictly better, else keep grid.
            if res.fun < best_neg - 1e-6:
                best_neg, best_w = float(res.fun), np.asarray(res.x, dtype=float)
                best_w = np.clip(best_w, 0, None)
                if best_w.sum() > 0:
                    best_w = best_w / best_w.sum()
    except Exception:
        pass

    return best_w, -best_neg  # return positive Spearman


# ---------------------------------------------------------------------------
# LOO
# ---------------------------------------------------------------------------

def leave_one_out(components: np.ndarray, gold: np.ndarray,
                  multi: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    For each held-out system i: fit weights on the other 18, then compute
    Spearman of AQS_w_i across all 19 systems vs gold across all 19.

    Returns
    -------
    weights_per_fold : (n, 3) — fitted weights per fold
    rhos_per_fold    : (n,)   — Spearman across all 19 with fold's weights
    """
    n = components.shape[0]
    weights_per_fold = np.zeros((n, 3), dtype=float)
    rhos_per_fold    = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w_i, _ = fit_weights(components[mask], gold[mask] if not multi else gold[mask, :], multi=multi)
        weights_per_fold[i] = w_i
        # Evaluate AQS_w_i on all 19 systems against gold (Recall@10 — the
        # primary single-metric objective; for multi we keep R@10 as the
        # held-out diagnostic since the user's headline question is "how
        # well do these weights rank by R@10").
        gold_eval = gold[:, 0] if multi else gold
        pred = aqs(w_i, components)
        rhos_per_fold[i] = spearmanr(pred, gold_eval).statistic
    return weights_per_fold, rhos_per_fold


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def components_alone_spearman(df: pd.DataFrame, gold_col: str) -> dict[str, float]:
    return {
        c: float(spearmanr(df[c], df[gold_col]).statistic)
        for c in COMPONENTS
    }


def main() -> None:
    df = load_system_table()
    print(f"Loaded {len(df)} systems")

    components = df[COMPONENTS].to_numpy(dtype=float)
    gold       = df[GOLD_PRIMARY].to_numpy(dtype=float)
    gold_multi = df[GOLD_MULTI].to_numpy(dtype=float)

    # --- 1) Sanity: each component alone vs R@10 ---
    components_alone = components_alone_spearman(df, GOLD_PRIMARY)
    print("\n[Sanity] Component-alone Spearman vs R@10:")
    for c, r in components_alone.items():
        print(f"  {c:10s} {r:+.4f}")

    # --- 2) Old 7:6:3 weights ---
    old_w = np.array([OLD_WEIGHTS[c] for c in COMPONENTS])
    old_pred = aqs(old_w, components)
    rho_old = float(spearmanr(old_pred, gold).statistic)
    print(f"\n[Old AQS 7:6:3] Spearman vs R@10 = {rho_old:+.4f}")

    # --- 3) Full-data fit (single-metric objective) ---
    w_full, rho_full = fit_weights(components, gold, multi=False)
    print(f"\n[Full-data fit, single-metric] weights = "
          f"{dict(zip(COMPONENTS, np.round(w_full, 4)))}  rho={rho_full:+.4f}")

    # --- 4) LOO single-metric ---
    w_loo, rho_loo = leave_one_out(components, gold, multi=False)
    rho_loo_mean = float(np.mean(rho_loo))
    w_loo_mean   = w_loo.mean(axis=0)
    w_loo_lo     = np.percentile(w_loo, 2.5,  axis=0)
    w_loo_hi     = np.percentile(w_loo, 97.5, axis=0)
    print(f"\n[LOO, single-metric] mean rho across 19 folds = {rho_loo_mean:+.4f}")
    print(f"  per-weight mean   {dict(zip(COMPONENTS, np.round(w_loo_mean, 4)))}")
    print(f"  per-weight 2.5%   {dict(zip(COMPONENTS, np.round(w_loo_lo, 4)))}")
    print(f"  per-weight 97.5%  {dict(zip(COMPONENTS, np.round(w_loo_hi, 4)))}")

    # --- 5) Multi-metric variant ---
    w_full_mm, rho_full_mm = fit_weights(components, gold_multi, multi=True)
    w_loo_mm, rho_loo_mm   = leave_one_out(components, gold_multi, multi=True)
    print(f"\n[Multi-metric (R@10, NDCG@10, MAP@10)] full weights = "
          f"{dict(zip(COMPONENTS, np.round(w_full_mm, 4)))}  "
          f"mean-rho_full = {rho_full_mm:+.4f}, mean rho_loo = {np.mean(rho_loo_mm):+.4f}")

    # --- 6) Persist JSON ---
    payload = {
        "n_systems":      int(len(df)),
        "components":     COMPONENTS,
        "gold_primary":   GOLD_PRIMARY,
        "gold_multi":     GOLD_MULTI,
        "old_weights":    OLD_WEIGHTS,

        "weights_full":   dict(zip(COMPONENTS, [float(x) for x in w_full])),
        "weights_loo": {
            "mean":    dict(zip(COMPONENTS, [float(x) for x in w_loo_mean])),
            "ci_low":  dict(zip(COMPONENTS, [float(x) for x in w_loo_lo])),
            "ci_high": dict(zip(COMPONENTS, [float(x) for x in w_loo_hi])),
            "per_fold": [
                {"held_out": df["experiment_id"].iloc[i],
                 "weights": dict(zip(COMPONENTS, [float(x) for x in w_loo[i]])),
                 "rho_full19_with_fold_weights": float(rho_loo[i])}
                for i in range(len(df))
            ],
        },
        "rho_in_sample":  float(rho_full),
        "rho_loo_mean":   rho_loo_mean,
        "rho_old_7_6_3":  rho_old,
        "components_alone": components_alone,

        "multi_metric": {
            "weights_full":     dict(zip(COMPONENTS, [float(x) for x in w_full_mm])),
            "rho_in_sample":    float(rho_full_mm),
            "rho_loo_mean":     float(np.mean(rho_loo_mm)),
            "weights_loo_mean": dict(zip(COMPONENTS, [float(x) for x in w_loo_mm.mean(axis=0)])),
        },

        "method": ("max-Spearman vs Recall@10, simplex grid (step=0.05) + "
                   "SLSQP refinement; LOO over 19 systems, per-fold weights "
                   "and Spearman of all-19 AQS recorded."),
    }
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[Wrote] {_OUT_JSON.relative_to(_ROOT).as_posix()}")

    # --- 7) Patch summary.csv: add T3/AQS_dd column (full-data weights) ---
    if _SUMMARY.exists():
        s = pd.read_csv(_SUMMARY)
        # Rebuild components column-aligned to summary order (in case they
        # differ — they shouldn't, both come from the same record dir)
        comp_lookup = {row["experiment_id"]: row[COMPONENTS].to_dict()
                       for _, row in df.iterrows()}
        def _aqs_dd(eid: str) -> float | None:
            comps = comp_lookup.get(eid)
            if comps is None:
                return None
            return float(sum(w_full[i] * comps[c] for i, c in enumerate(COMPONENTS)))
        s["T3/AQS_dd"] = s["experiment_id"].map(_aqs_dd)
        # Preserve column order: insert T3/AQS_dd right after T3/AQS
        cols = list(s.columns)
        if "T3/AQS_dd" in cols and "T3/AQS" in cols:
            cols.remove("T3/AQS_dd")
            idx = cols.index("T3/AQS") + 1
            cols.insert(idx, "T3/AQS_dd")
            s = s[cols]
        s.to_csv(_SUMMARY, index=False)
        print(f"[Patched] {_SUMMARY.relative_to(_ROOT).as_posix()} (added T3/AQS_dd)")


if __name__ == "__main__":
    main()
