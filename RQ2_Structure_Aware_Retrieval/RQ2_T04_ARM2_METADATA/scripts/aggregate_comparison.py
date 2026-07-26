"""Aggregate the five per-stem T03-vs-T04 comparison CSVs into a single view.

Reads:
  RQ2_T04_ARM2_METADATA/data/comparison_t03_vs_t04_<stem>.csv  (5 files)

Writes:
  analysis/comparison_all_stems_v4.csv     long: (stem, method, metric, value)
  analysis/comparison_summary_v4.csv       wide: one block per metric, rows=method, cols=stem (+ mean/median)
  analysis/comparison_summary_v4.md        markdown analogue with bold per-column best,
                                           T04-vs-T03 wins matrix, best-T04-variant ranking.

Latency is dropped from every output (kept in source CSV for anomaly debug only).
Cw/R@10 and Cw/R@100 are carried only when populated (typically doc 9 only).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_DATA = _HERE.parent / "data"           # per-stem source CSVs + per-query JSONs live here
_ANALYSIS = _HERE.parent / "analysis"   # aggregated outputs land here

STEMS = [
    ("1967_10_10_1967101056", 5, "Code Judiciaire (smaller)"),
    ("1867_06_08_1867060850", 6, "Code Penal"),
    ("1804_03_21_1804032150", 9, "Code Civil"),
    ("2003_07_17_2013A31614", 8, "Loi (rentals)"),
    ("1967_10_10_1967101055", 7, "Code Judiciaire (larger)"),
]

# Analysis metrics — order matters for output layout.
ANALYSIS_METRICS = ["R@10", "R@100", "MRR@10", "nDCG@10", "Cw/R@10", "Cw/R@100"]
DROP_COLS = ["latency_ms"]

METHOD_ORDER = [
    "arm1_naive",
    "arm2_metadata_node_raw",
    "arm2_metadata_node_enriched",
    "arm2_metadata_node_summary",
    "arm2_metadata_node_full",
    "arm2_metadata_article_raw",
    "arm2_metadata_article_full",
]

T04_METHODS = [m for m in METHOD_ORDER if m.startswith("arm2_")]


def _load_one(stem: str) -> pd.DataFrame:
    path = _DATA / f"comparison_t03_vs_t04_{stem}.csv"
    df = pd.read_csv(path)
    # Drop any `best:*` annotation columns (Phase 1 emitted these; we recompute).
    df = df[[c for c in df.columns if not c.startswith("best:")]]
    # Drop latency from every aggregation.
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)
    df["stem"] = stem
    return df


def _load_all() -> pd.DataFrame:
    frames = [_load_one(s) for s, _, _ in STEMS]
    df = pd.concat(frames, ignore_index=True, sort=False)
    # Keep only known method rows in canonical order.
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["stem", "method"]).reset_index(drop=True)
    return df


def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Long format: (stem, method, metric, value). NaN metrics are dropped."""
    id_cols = ["stem", "method", "n_queries"]
    metric_cols = [c for c in df.columns if c in ANALYSIS_METRICS]
    long = df.melt(
        id_vars=id_cols, value_vars=metric_cols, var_name="metric", value_name="value"
    )
    long = long.dropna(subset=["value"]).reset_index(drop=True)
    return long


def _wide_block(df: pd.DataFrame, metric: str) -> pd.DataFrame | None:
    """Build a method x stem pivot for one metric. Returns None if metric absent."""
    if metric not in df.columns:
        return None
    sub = df[["stem", "method", metric]].dropna(subset=[metric])
    if sub.empty:
        return None
    pivot = sub.pivot(index="method", columns="stem", values=metric)
    # Enforce method order.
    pivot = pivot.reindex([m for m in METHOD_ORDER if m in pivot.index])
    # Enforce stem column order.
    stem_order = [s for s, _, _ in STEMS if s in pivot.columns]
    pivot = pivot[stem_order]
    pivot["mean"] = pivot.mean(axis=1, skipna=True)
    pivot["median"] = pivot.median(axis=1, skipna=True)
    return pivot


def _format_md_table(pivot: pd.DataFrame, metric: str) -> str:
    """Markdown table for one metric, bolding the per-column best."""
    stem_cols = [c for c in pivot.columns if c not in ("mean", "median")]
    # Column header — short doc labels for readability.
    short_label = {s: f"doc {d}" for s, d, _ in STEMS}
    head_cells = ["method"] + [short_label.get(s, s) for s in stem_cols] + ["mean", "median"]
    sep = ["---"] + ["---:"] * (len(stem_cols) + 2)

    # Find per-stem best for bolding (ties: bold all winners).
    per_col_best = {}
    for s in stem_cols:
        col = pivot[s].dropna()
        if col.empty:
            continue
        best_val = col.max()
        per_col_best[s] = best_val

    lines = ["| " + " | ".join(head_cells) + " |", "| " + " | ".join(sep) + " |"]
    for method, row in pivot.iterrows():
        cells = [method]
        for s in stem_cols:
            v = row.get(s)
            if pd.isna(v):
                cells.append("—")
            else:
                txt = f"{v:.4f}"
                if s in per_col_best and abs(v - per_col_best[s]) < 1e-12:
                    txt = f"**{txt}**"
                cells.append(txt)
        cells.append("—" if pd.isna(row["mean"]) else f"{row['mean']:.4f}")
        cells.append("—" if pd.isna(row["median"]) else f"{row['median']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _wins_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """For each (stem, metric), record the winning method.

    Returns wide table: rows=method, cols=metric, values=# stems where method wins.
    """
    rows = []
    for s, _, _ in STEMS:
        sub_stem = df[df["stem"] == s]
        for metric in ANALYSIS_METRICS:
            if metric not in sub_stem.columns:
                continue
            col = sub_stem[["method", metric]].dropna(subset=[metric])
            if col.empty:
                continue
            best_val = col[metric].max()
            winners = col[col[metric].eq(best_val)]["method"].tolist()
            for w in winners:
                rows.append({"stem": s, "metric": metric, "winner": w, "share": 1.0 / len(winners)})
    wins = pd.DataFrame(rows)
    if wins.empty:
        return pd.DataFrame()
    pivot = wins.pivot_table(
        index="winner", columns="metric", values="share", aggfunc="sum"
    ).fillna(0.0)
    # Re-order rows and metric cols.
    method_idx = [m for m in METHOD_ORDER if m in pivot.index]
    pivot = pivot.reindex(method_idx)
    metric_idx = [m for m in ANALYSIS_METRICS if m in pivot.columns]
    pivot = pivot[metric_idx]
    return pivot


def _t04_vs_t03(df: pd.DataFrame) -> pd.DataFrame:
    """For each metric: # stems where best T04 variant beats T03."""
    rows = []
    for metric in ANALYSIS_METRICS:
        if metric not in df.columns:
            continue
        rec = {"metric": metric, "n_stems": 0, "t04_wins": 0, "t03_wins": 0, "tie": 0}
        for s, _, _ in STEMS:
            sub = df[(df["stem"] == s) & df[metric].notna()]
            if sub.empty:
                continue
            t03_row = sub[sub["method"] == "arm1_naive"]
            t04_rows = sub[sub["method"].isin(T04_METHODS)]
            if t03_row.empty or t04_rows.empty:
                continue
            t03_val = float(t03_row[metric].iloc[0])
            t04_best = float(t04_rows[metric].max())
            rec["n_stems"] += 1
            if t04_best > t03_val + 1e-9:
                rec["t04_wins"] += 1
            elif t03_val > t04_best + 1e-9:
                rec["t03_wins"] += 1
            else:
                rec["tie"] += 1
        rows.append(rec)
    return pd.DataFrame(rows)


def _best_t04_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Mean rank of each T04 variant across (stem x metric). Lowest = best overall."""
    rank_rows = []
    for s, _, _ in STEMS:
        for metric in ANALYSIS_METRICS:
            if metric not in df.columns:
                continue
            sub = df[(df["stem"] == s) & df["method"].isin(T04_METHODS)][["method", metric]].dropna(
                subset=[metric]
            )
            if sub.empty:
                continue
            # Higher value = better rank (rank 1).
            sub = sub.copy()
            sub["rank"] = sub[metric].rank(ascending=False, method="min")
            for _, r in sub.iterrows():
                rank_rows.append(
                    {"method": r["method"], "stem": s, "metric": metric, "rank": r["rank"]}
                )
    if not rank_rows:
        return pd.DataFrame()
    ranks = pd.DataFrame(rank_rows)
    summary = (
        ranks.groupby("method")
        .agg(mean_rank=("rank", "mean"), n_cells=("rank", "size"))
        .sort_values("mean_rank")
        .reset_index()
    )
    return summary


def _stem_header_md() -> str:
    rows = ["| Stem | doc_id | Title |", "|---|---:|---|"]
    for s, d, t in STEMS:
        rows.append(f"| `{s}` | {d} | {t} |")
    return "\n".join(rows)


def main() -> None:
    df = _load_all()

    _ANALYSIS.mkdir(parents=True, exist_ok=True)

    # --- Artefact 1: long format
    long = _to_long(df)
    long_path = _ANALYSIS / "comparison_all_stems_v4.csv"
    long.to_csv(long_path, index=False)
    print(f"wrote {long_path}  ({len(long)} rows)")

    # --- Artefact 2: wide machine-readable CSV (one block per metric, stacked)
    wide_blocks: list[tuple[str, pd.DataFrame]] = []
    for metric in ANALYSIS_METRICS:
        block = _wide_block(df, metric)
        if block is not None:
            wide_blocks.append((metric, block))

    wide_path = _ANALYSIS / "comparison_summary_v4.csv"
    parts = []
    for metric, block in wide_blocks:
        b = block.copy()
        b.insert(0, "metric", metric)
        b = b.reset_index().rename(columns={"index": "method"})
        parts.append(b)
    wide_csv = pd.concat(parts, ignore_index=True)
    wide_csv.to_csv(wide_path, index=False)
    print(f"wrote {wide_path}  ({len(wide_csv)} rows)")

    # --- Artefact 3: markdown summary
    # The file may carry a hand-written preamble (e.g. "Headline findings").
    # We preserve everything before the AUTO-BEGIN sentinel so re-running the
    # aggregator does not clobber it.
    md_lines = [
        "<!-- AUTO-BEGIN: do not edit below; regenerated by aggregate_comparison.py -->",
        "",
        "## Detailed tables",
        "",
        "Source: per-stem comparison CSVs at `../data/comparison_t03_vs_t04_<stem>.csv`",
        "(LINKER_VERSION = 4, Phase 1 v4 sweep).",
        "",
        "### Stems",
        "",
        _stem_header_md(),
        "",
        "Analysis metrics reported here: "
        + ", ".join(f"`{m}`" for m in ANALYSIS_METRICS)
        + ". `latency_ms` is intentionally excluded.",
        "",
    ]

    for metric, block in wide_blocks:
        md_lines.append(f"### {metric}")
        md_lines.append("")
        md_lines.append(_format_md_table(block, metric))
        md_lines.append("")
        md_lines.append("Bold = per-stem best across all 7 methods.")
        md_lines.append("")

    # Wins matrix
    md_lines.append("### Wins by (stem x metric) — how many stems each method wins")
    md_lines.append("")
    wins = _wins_matrix(df)
    if not wins.empty:
        head = ["method"] + list(wins.columns)
        sep = ["---"] + ["---:"] * len(wins.columns)
        md_lines.append("| " + " | ".join(head) + " |")
        md_lines.append("| " + " | ".join(sep) + " |")
        for method, row in wins.iterrows():
            cells = [method] + [f"{row[c]:.2f}" if row[c] else "0" for c in wins.columns]
            md_lines.append("| " + " | ".join(cells) + " |")
        md_lines.append("")
        md_lines.append("Ties split the win equally across tied methods.")
    md_lines.append("")

    # T04-vs-T03 head-to-head
    md_lines.append("### T04 (best variant) vs T03 — per metric")
    md_lines.append("")
    h2h = _t04_vs_t03(df)
    if not h2h.empty:
        md_lines.append("| metric | n_stems | T04 wins | T03 wins | tie |")
        md_lines.append("|---|---:|---:|---:|---:|")
        for _, r in h2h.iterrows():
            md_lines.append(
                f"| {r['metric']} | {int(r['n_stems'])} | {int(r['t04_wins'])} | "
                f"{int(r['t03_wins'])} | {int(r['tie'])} |"
            )
    md_lines.append("")

    # Best T04 variant overall
    md_lines.append("### Best T04 variant overall (lowest mean rank)")
    md_lines.append("")
    bv = _best_t04_variant(df)
    if not bv.empty:
        md_lines.append("| method | mean_rank | n_cells |")
        md_lines.append("|---|---:|---:|")
        for _, r in bv.iterrows():
            md_lines.append(f"| {r['method']} | {r['mean_rank']:.3f} | {int(r['n_cells'])} |")
    md_lines.append("")
    md_lines.append("Rank computed per (stem, metric); ties get the min rank. Lowest mean wins.")
    md_lines.append("")

    md_path = _ANALYSIS / "comparison_summary_v4.md"
    sentinel = "<!-- AUTO-BEGIN: do not edit below; regenerated by aggregate_comparison.py -->"
    preamble = ""
    if md_path.exists():
        existing = md_path.read_text(encoding="utf-8")
        if sentinel in existing:
            preamble = existing.split(sentinel, 1)[0]
    auto_block = "\n".join(md_lines)
    md_path.write_text(preamble + auto_block, encoding="utf-8")
    print(f"wrote {md_path}  (auto block: {len(md_lines)} lines; preamble preserved: {bool(preamble)})")


if __name__ == "__main__":
    main()
