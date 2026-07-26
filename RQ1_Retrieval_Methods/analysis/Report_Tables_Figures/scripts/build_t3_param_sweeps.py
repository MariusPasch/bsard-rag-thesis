"""
Build the Tier 3 parameter-sweep figure.

Output: Report/figures/fig_rq1_t3_param_sweeps.{pdf,png}

What's drawn
------------
Three panels horizontally, one per hybrid sub-family:
  Panel 1 -- RRF dampening k    (x = k     in {30, 60, 120})
  Panel 2 -- Linear alpha       (x = alpha in {0.1, ..., 0.9})
  Panel 3 -- SGDR pool size K   (x = K     in {1000, 2000, 5000})

Each panel shows two lines:
  - Recall@10  (solid green, diamond markers) -- T3 primary metric
  - Recall@100 (dashed green, circle markers) -- recall ceiling

The family canonical (best R@10 within the family) is marked with a star
above the corresponding x-value.

Reads parameter values from the result-JSON `hyperparameters` blocks, so the
script does not duplicate the sweep grid.

Usage:
  .venv/Scripts/python Report/scripts/build_t3_param_sweeps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPORT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPORT_DIR))

from load_results import load_records
from thesis_style import PALETTE, rcparams_report, save_figure

OUTPUT_NAME = "fig_rq1_t3_param_sweeps"

# (family_label, hyperparameter_key, x_axis_label, panel_title)
SUBFAMILIES = [
    ("rrf",           "rrf_k",        r"RRF dampening $k$",
     "RRF (panel 1)"),
    ("linear",        "alpha",        r"Linear interpolation weight $\alpha$",
     "Linear (panel 2)"),
    ("sgdr",          "K",            r"SGDR candidate-pool size $K$",
     "SGDR (panel 3)"),
]


def _extract_param(rec, family: str, key: str) -> float | None:
    """
    Pull a single parameter value from the result JSON's hyperparameters block.

    Hyperparameters differ slightly by family:
      - RRF      -> hyperparameters.rrf_k
      - linear   -> hyperparameters.alpha (or .linear_alpha)
      - SGDR     -> hyperparameters.K     (or .sgdr_K)
    """
    h = rec.hyperparameters or {}
    if key in h:
        return float(h[key])
    # Try common alternative spellings.
    for alt in (f"{family}_{key}", f"{family}_{key.lower()}", key.lower()):
        if alt in h:
            return float(h[alt])
    return None


def _classify(rec) -> str | None:
    """Return one of {'rrf', 'linear', 'sgdr'} based on experiment_id."""
    eid = rec.experiment_id
    if "rrf" in eid:
        return "rrf"
    if "linear" in eid:
        return "linear"
    if "sgdr" in eid:
        return "sgdr"
    return None


def main() -> None:
    records = load_records()
    t3 = [r for t, r in records if t == "T3"]
    if not t3:
        raise SystemExit("No Tier 3 result records found.")

    # Group records by family and pull (param, R@10, R@100) tuples.
    by_family: dict[str, list[tuple[float, float, float]]] = {
        "rrf": [], "linear": [], "sgdr": [],
    }
    for rec in t3:
        fam = _classify(rec)
        if fam is None:
            print(f"[warn] unclassified T3 record: {rec.experiment_id}")
            continue
        key = next(k for f, k, *_ in SUBFAMILIES if f == fam)
        x = _extract_param(rec, fam, key)
        if x is None:
            # Fall back to parsing the experiment_id (e.g. hybrid_rrf_k60_test
            # -> 60, hybrid_linear_alpha_0.3_test -> 0.3).
            eid = rec.experiment_id
            try:
                if fam == "rrf":
                    x = float(eid.split("_k")[1].split("_")[0])
                elif fam == "linear":
                    x = float(eid.split("_alpha_")[1].split("_")[0])
                elif fam == "sgdr":
                    x = float(eid.split("_k")[1].split("_")[0])
            except Exception:
                print(f"[warn] couldn't extract {key} for {eid}")
                continue
        m = rec.metrics
        r10  = float(m.get("Recall@10",  float("nan")))
        r100 = float(m.get("Recall@100", float("nan")))
        by_family[fam].append((x, r10, r100))

    # Sort points within each family by x ascending so the line plots cleanly.
    for fam in by_family:
        by_family[fam].sort(key=lambda t: t[0])

    rcparams_report()
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))

    color_r10  = PALETTE["green"]
    color_r100 = PALETTE["green"]

    for ax, (fam, key, xlabel, title) in zip(axes, SUBFAMILIES):
        pts = by_family[fam]
        if not pts:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            continue

        xs   = [t[0] for t in pts]
        r10s = [t[1] for t in pts]
        r100s = [t[2] for t in pts]

        ax.plot(xs, r10s, color=color_r10, marker="D", linestyle="-",
                linewidth=1.4, markersize=4.8, label=r"Recall@10")
        ax.plot(xs, r100s, color=color_r100, marker="o", linestyle="--",
                linewidth=1.2, markersize=4.4, alpha=0.75,
                label=r"Recall@100")

        # Star above the canonical x (= argmax R@10 within family).
        best_idx = int(np.argmax(r10s))
        ax.plot(xs[best_idx], r10s[best_idx] + 0.025,
                marker="*", markersize=10, color=PALETTE["black"],
                linestyle="none", zorder=10, clip_on=False)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(fam.upper(), fontsize=10, pad=4)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(0.20, 0.75)

    axes[0].set_ylabel(r"Recall (test, $n=222$)")
    axes[0].legend(loc="lower right", fontsize=8, frameon=False,
                   handlelength=2.2, handletextpad=0.6)

    fig.tight_layout()
    paths = save_figure(fig, OUTPUT_NAME)
    plt.close(fig)
    for p in paths:
        print(f"[done] Wrote {p.relative_to(REPORT_DIR.parent)}")


if __name__ == "__main__":
    main()
