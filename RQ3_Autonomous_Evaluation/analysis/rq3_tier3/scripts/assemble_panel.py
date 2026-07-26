"""
analysis/rq3_tier3/scripts/assemble_panel.py
─────────────────────────────────────────────
Assemble the canonical (system × query × doc) panel for RQ3 Tier 3 analysis.

Reads, per family (sparse / dense / hybrid / agentic) under RQ1's
`output/results/<family>/`:
  * <exp>_test.json (sparse) or <exp>_zeroshot_test.json (dense), with
    `subset_metrics.metrics` populated by `compute_subset_metrics.py` and the
    Tier 3 run script.
  * tier3_per_query/<exp>.json sidecars produced by the patched
    run_<family>_tier3.py scripts.

Joins on:
  * evaluation/data/tier3_subset.json   — 48 questions + strata + gold articles
  * evaluation/data/query_strata.json   — bm25_score per question
  * output/bsard_articles_dedup.parquet — article → law_code

Outputs (under <analysis>/rq3_tier3/):
  data/<family>_panel.parquet       — long-format, one row per (system, qid, doc at rank ≤ 10)
  data/panel_combined.parquet        — concat of all populated families
  data/strata_summary.json           — descriptive n per cell / per axis
  tables/system_summary.tsv          — per-system aggregate (R@k, NDCG, evaluator means)

The script is **family-agnostic and tolerant of missing data**:
  * If a family's results dir does not exist → skip it.
  * If `subset_metrics` is missing for a system → skip that system, log it.
  * If a sidecar is missing → emit only the system summary row; skip per-(q,d)
    rows for that system (per-query columns would be all-null anyway).

Usage (from project root of RQ3_Autonomous_Evaluation)
─────────────────────────────────────────────────────────
  .venv/Scripts/python analysis/rq3_tier3/scripts/assemble_panel.py
  .venv/Scripts/python analysis/rq3_tier3/scripts/assemble_panel.py \\
      --rq1-root "../RQ1_Retrieval_Methods" --families sparse,dense

Plan: analysis/rq3_tier3/README.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Family configuration
# ---------------------------------------------------------------------------

_FAMILY_SPECS: dict[str, tuple[str, str]] = {
    # family -> (results-subdir-under-RQ1/output/results, exp_id suffix on JSON)
    "sparse":  ("sparse_retrieval",  "_test"),
    "dense":   ("dense_retrieval",   "_zeroshot_test"),
    "hybrid":  ("hybrid_retrieval",  "_test"),
    "agentic": ("agentic_retrieval", "_test"),
}

# How many top-k retrieved docs we keep per query in the panel.
# Tier 3 evaluation runs at k=10 — that's the document set the LLM judges saw.
_PANEL_TOP_K = 10


# ---------------------------------------------------------------------------
# Reference-data loaders
# ---------------------------------------------------------------------------

def load_subset(path: Path) -> dict[str, dict[str, Any]]:
    """{qid_str: {question_text, gold_article_ids: set[int], strata: {...}}}"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(q["question_id"]): {
            "question_text":     q["question_text"],
            "gold_article_ids":  set(q["relevant_article_ids"]),
            "n_gold":            int(q["n_relevant_articles"]),
            "strata":            q["strata"],
        }
        for q in raw["questions"]
    }


def load_strata(path: Path) -> dict[str, dict[str, Any]]:
    return {str(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def load_article_law_codes(parquet_path: Path) -> dict[int, str]:
    df = pd.read_parquet(parquet_path, columns=["article_id", "law_code"])
    return dict(zip(df["article_id"].astype(int), df["law_code"].astype(str)))


def derive_question_law_code(
    gold_article_ids: set[int],
    article_law_codes: dict[int, str],
) -> str:
    """
    A single law_code label per question = mode of gold articles' law_codes,
    breaking ties alphabetically.  Returns "unknown" if no gold article has a
    known law_code.
    """
    codes = [article_law_codes[a] for a in gold_article_ids if a in article_law_codes]
    if not codes:
        return "unknown"
    counter = Counter(codes)
    top = counter.most_common()
    max_count = top[0][1]
    tied = sorted(c for c, n in top if n == max_count)
    return tied[0]


# ---------------------------------------------------------------------------
# Per-system extraction
# ---------------------------------------------------------------------------

@dataclass
class SystemRecord:
    family: str
    system: str
    exp_id: str
    subset_metrics: dict[str, float]
    sidecar: Optional[dict[str, Any]]


def _short_name(exp_id: str, suffix: str) -> str:
    """Strip the family suffix from the experiment_id to get the short name."""
    return exp_id[:-len(suffix)] if suffix and exp_id.endswith(suffix) else exp_id


def load_system_records(
    family: str,
    results_dir: Path,
    sidecar_dir: Path,
    suffix: str,
) -> list[SystemRecord]:
    pattern = f"*{suffix}.json"
    records: list[SystemRecord] = []
    for fpath in sorted(results_dir.glob(pattern)):
        # Skip the dense "_sel_*" selection-run files
        if fpath.name.startswith("_sel_"):
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [skip] {fpath.name}: cannot parse JSON ({exc})")
            continue

        exp_id = data.get("experiment_id", fpath.stem)
        sm = data.get("subset_metrics", {}).get("metrics")
        if not sm:
            print(f"  [skip] {fpath.name}: no subset_metrics.metrics block "
                  f"— run compute_subset_metrics.py first")
            continue

        sidecar_path = sidecar_dir / f"{exp_id}.json"
        sidecar: Optional[dict[str, Any]] = None
        if sidecar_path.exists():
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"  [warn] {fpath.name}: sidecar parse error ({exc})")
                sidecar = None
        else:
            print(f"  [warn] {fpath.name}: no sidecar at "
                  f"{sidecar_path.relative_to(sidecar_dir.parent.parent.parent)}"
                  " — system row only, no per-(q,d) data")

        records.append(SystemRecord(
            family=family,
            system=_short_name(exp_id, suffix),
            exp_id=exp_id,
            subset_metrics=sm,
            sidecar=sidecar,
        ))
    return records


# ---------------------------------------------------------------------------
# Panel + summary builders
# ---------------------------------------------------------------------------

def build_per_qd_rows(
    rec: SystemRecord,
    subset: dict[str, dict[str, Any]],
    article_law_codes: dict[int, str],
) -> list[dict[str, Any]]:
    """
    Emit one row per (qid, rank ≤ _PANEL_TOP_K) for this system.  Returns []
    if the sidecar is missing or has no `ranks`.
    """
    if rec.sidecar is None or not rec.sidecar.get("ranks"):
        return []

    ranks    = rec.sidecar["ranks"]                      # {qid: [doc_id, ...]}
    umbrela  = rec.sidecar.get("umbrela")  or {}
    erag     = rec.sidecar.get("erag")     or {}
    ragas_wa = rec.sidecar.get("ragas_wa") or {}

    rows: list[dict[str, Any]] = []
    for qid, doc_ids in ranks.items():
        q = subset.get(qid)
        if q is None:
            # Sidecar covers a query outside the canonical 48-question subset
            # — skip rather than introduce off-subset rows.
            continue
        gold_ids   = q["gold_article_ids"]
        strata     = q["strata"]
        bm25_score = strata.get("bm25_score")
        law_code   = derive_question_law_code(gold_ids, article_law_codes)
        ragas_q    = ragas_wa.get(qid)

        for rank, doc_id in enumerate(doc_ids[:_PANEL_TOP_K], start=1):
            try:
                doc_id_int = int(doc_id)
            except (TypeError, ValueError):
                doc_id_int = -1
            umbrela_grade = (umbrela.get(qid) or {}).get(doc_id)
            erag_score    = (erag.get(qid)    or {}).get(doc_id)
            rows.append({
                "family":               rec.family,
                "system":               rec.system,
                "question_id":          int(qid),
                "rank":                 rank,
                "doc_id":               doc_id_int,
                "bsard_relevant":       int(doc_id_int in gold_ids),
                "umbrela_grade":        int(umbrela_grade) if umbrela_grade is not None else pd.NA,
                "erag_score":           int(erag_score) if erag_score is not None else pd.NA,
                "ragas_wa_query_score": float(ragas_q) if ragas_q is not None else pd.NA,
                "bm25_score":           float(bm25_score) if bm25_score is not None else pd.NA,
                "article_count":        strata.get("article_count"),
                "lex_align":            strata.get("lex_align"),
                "cross_ref":            strata.get("cross_ref"),
                "law_code":             law_code,
                "n_gold":               q["n_gold"],
            })
    return rows


def build_system_summary_row(rec: SystemRecord) -> dict[str, Any]:
    sm = rec.subset_metrics
    n_queries = sum(1 for _ in (rec.sidecar or {}).get("ranks", {}))
    return {
        "family":                  rec.family,
        "system":                  rec.system,
        "experiment_id":           rec.exp_id,
        "n_queries_with_sidecar":  n_queries,
        "Recall@10":               sm.get("Recall@10"),
        "Recall@100":              sm.get("Recall@100"),
        "NDCG@10":                 sm.get("NDCG@10"),
        "MAP":                     sm.get("MAP"),
        "MAP@100":                 sm.get("MAP@100"),
        "T3_umbrela_mean":         sm.get("T3/umbrela/mean"),
        "T3_umbrela_mean_grade":   sm.get("T3/umbrela/mean_grade"),
        "T3_erag_mean":            sm.get("T3/erag/mean"),
        "T3_ragas_wa_mean":        sm.get("T3/ragas_wa/mean"),
        # T2-umbrela bridge — UMBRELA grades treated as qrels and run through Tier 2
        "T2_umbrela_NDCG@10":      sm.get("T2-umbrela/P2/NDCG@10"),
        "T2_umbrela_MAP@100":      sm.get("T2-umbrela/P2/MAP@100"),
    }


def write_strata_summary(
    subset: dict[str, dict[str, Any]],
    article_law_codes: dict[int, str],
    out_path: Path,
) -> None:
    cells: dict[str, int] = {}
    axes: dict[str, dict[str, int]] = {
        "article_count": {},
        "lex_align":     {},
        "cross_ref":     {},
        "law_code":      {},
    }
    for q in subset.values():
        s = q["strata"]
        cell_key = f"({s.get('article_count')}, {s.get('lex_align')}, {s.get('cross_ref')})"
        cells[cell_key] = cells.get(cell_key, 0) + 1
        for axis in ("article_count", "lex_align", "cross_ref"):
            axes[axis][s.get(axis)] = axes[axis].get(s.get(axis), 0) + 1
        lc = derive_question_law_code(q["gold_article_ids"], article_law_codes)
        axes["law_code"][lc] = axes["law_code"].get(lc, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"total": len(subset), "axes": axes, "cells": cells},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve()
    rq3_root = here.parent.parent          # analysis/rq3_tier3/
    default_rq1_root = rq3_root.parent.parent.parent / "RQ1_Retrieval_Methods"

    parser = argparse.ArgumentParser(
        description="Assemble RQ3 Tier 3 analysis panel from per-family "
                    "result JSONs and per-query sidecars."
    )
    parser.add_argument(
        "--rq1-root", type=Path, default=default_rq1_root,
        help=f"Root of the RQ1 project (default: {default_rq1_root})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=rq3_root,
        help=f"Output root for data/ and tables/ (default: {rq3_root})",
    )
    parser.add_argument(
        "--families", type=str, default="sparse,dense",
        help="Comma-separated family names to assemble (default: sparse,dense)",
    )
    args = parser.parse_args()

    rq1_root = args.rq1_root.resolve()
    out_dir  = args.out_dir.resolve()
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    print("=" * 70)
    print("RQ3 Tier 3 — assemble panel")
    print("=" * 70)
    print(f"RQ1 root  : {rq1_root}")
    print(f"Out dir   : {out_dir}")
    print(f"Families  : {families}\n")

    # ── Reference data ───────────────────────────────────────────────────────
    subset_path = rq1_root / "evaluation" / "data" / "tier3_subset.json"
    strata_path = rq1_root / "evaluation" / "data" / "query_strata.json"
    parquet_path = rq1_root / "output" / "bsard_articles_dedup.parquet"

    for p in (subset_path, strata_path, parquet_path):
        if not p.exists():
            print(f"[ERROR] required reference file not found: {p}")
            sys.exit(1)

    print(f"Loading subset      : {subset_path.relative_to(rq1_root)}")
    subset = load_subset(subset_path)
    print(f"  {len(subset)} questions")

    print(f"Loading strata      : {strata_path.relative_to(rq1_root)}")
    _strata_full = load_strata(strata_path)
    print(f"  {len(_strata_full)} strata entries (full test set)")

    print(f"Loading law codes   : {parquet_path.relative_to(rq1_root)}")
    article_law_codes = load_article_law_codes(parquet_path)
    print(f"  {len(article_law_codes)} articles\n")

    # ── Per-family assembly ──────────────────────────────────────────────────
    family_panels: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for family in families:
        if family not in _FAMILY_SPECS:
            print(f"[skip] {family}: unknown family")
            continue
        results_subdir, suffix = _FAMILY_SPECS[family]
        results_dir = rq1_root / "output" / "results" / results_subdir
        sidecar_dir = results_dir / "tier3_per_query"

        print(f"-- {family}")
        if not results_dir.exists():
            print(f"  [skip] no results dir at {results_dir}\n")
            continue

        records = load_system_records(family, results_dir, sidecar_dir, suffix)
        print(f"  {len(records)} system(s) with subset_metrics")

        all_rows: list[dict[str, Any]] = []
        for rec in records:
            rows = build_per_qd_rows(rec, subset, article_law_codes)
            all_rows.extend(rows)
            summary_rows.append(build_system_summary_row(rec))

        if all_rows:
            df = pd.DataFrame(all_rows)
            family_panels[family] = df
            data_dir = out_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            panel_path = data_dir / f"{family}_panel.parquet"
            df.to_parquet(panel_path, index=False)
            print(f"  wrote {panel_path.relative_to(out_dir)}: "
                  f"{len(df):,} rows ({df['system'].nunique()} systems × "
                  f"{df['question_id'].nunique()} questions × top-{_PANEL_TOP_K})")
        else:
            print(f"  no per-(q,d) rows assembled for {family} "
                  f"(sidecars missing or empty)")
        print()

    # ── Combined panel ───────────────────────────────────────────────────────
    if family_panels:
        combined = pd.concat(family_panels.values(), ignore_index=True)
        combined_path = out_dir / "data" / "panel_combined.parquet"
        combined.to_parquet(combined_path, index=False)
        print(f"Combined panel: {combined_path.relative_to(out_dir)}: "
              f"{len(combined):,} rows  ({combined['family'].nunique()} families × "
              f"{combined['system'].nunique()} systems)")
    else:
        print("No family panels assembled — combined panel not written.")

    # ── System summary ───────────────────────────────────────────────────────
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        tables_dir = out_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        summary_path = tables_dir / "system_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"System summary: {summary_path.relative_to(out_dir)}: "
              f"{len(summary_df)} systems")
    else:
        print("No system summary rows — table not written.")

    # ── Strata summary ───────────────────────────────────────────────────────
    strata_summary_path = out_dir / "data" / "strata_summary.json"
    write_strata_summary(subset, article_law_codes, strata_summary_path)
    print(f"Strata summary: {strata_summary_path.relative_to(out_dir)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
