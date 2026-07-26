"""Analyze dense-only retrieval performance on the curated 5 PDFs.

Evaluates against the **full BSARD question set** (1108 questions = the union
of the BSARD-shipped train and test splits). The split is purely a
benchmark-side convention from BSARD; no training is performed in RQ2 and
keeping the splits separate in the output table tends to confuse readers, so
they are combined here.

For each PDF:
  - Re-rank pool candidates by ``dense_score`` (FAISS only; sparse/fused
    ignored).
  - Aggregate chunks → bsard_ids by max dense score (matches
    ``adapter.to_bsard_run`` for the score-only case).
  - Compute T1/* over linked questions vs full GT.
  - Compute E/* (drift-aware) over evaluable questions vs ``gt_in_pdf``.

Outputs (default ``RQ2_T07_EVALUATION/analysis/``):
  - ``dense_retrieval_5pdf_<UTC-stamp>.csv``  — one row per PDF + aggregate
  - ``dense_retrieval_5pdf_<UTC-stamp>.md``   — short data-driven analysis
  - Markdown table to stdout

Substrate must be drift-aware. Sanity-checks abort with a clear error if any
inconsistency is detected. Use the T03 venv (PyMuPDF is required transitively).

Usage (from RQ2_T07_EVALUATION/ or RQ2_T03_ARM1_NAIVE/):
    python scripts/analyze_dense_retrieval.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

for _sibling in _REPO.glob("RQ2_T0*/src"):
    s = str(_sibling)
    if s not in sys.path:
        sys.path.insert(0, s)

from evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k  # noqa: E402
from evaluation.projection import ProjectionRow  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

PDFS = [
    "1804_03_21_1804032150",
    "1867_06_08_1867060850",
    "1967_10_10_1967101055",
    "1967_10_10_1967101056",
    "2003_07_17_2013A31614",
]

# BSARD ships its 1108 questions split into a train+test pair purely for
# benchmark convention. RQ2 does no training, so the split is irrelevant here
# — both files are loaded and merged into a single 1108-question evaluation
# set per PDF. The hashes are the canonical qset_hashes already on disk.
QSET_HASHES = {
    "bsard_train": "483beb56a47b",  # 886 Q
    "bsard_test":  "50744bfce1d0",  # 222 Q
}

DEFAULT_T03_CACHE_ROOT = _REPO / "RQ2_T03_ARM1_NAIVE" / "data"
DEFAULT_OUTPUT_DIR = _REPO / "RQ2_T07_EVALUATION" / "analysis"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _config_hash_for(doc_id: str, cache_root: Path) -> str:
    configs_dir = cache_root / doc_id / "configs"
    if not configs_dir.exists():
        raise FileNotFoundError(f"missing configs dir: {configs_dir}")
    subs = [p for p in configs_dir.iterdir() if p.is_dir()]
    if len(subs) != 1:
        raise RuntimeError(
            f"{doc_id}: expected exactly 1 config_hash dir, found {len(subs)} "
            f"({[p.name for p in subs]})"
        )
    return subs[0].name


def _load_projection(
    doc_id: str, qset_hash: str, cache_root: Path,
) -> tuple[dict, list[ProjectionRow]]:
    p = cache_root / doc_id / "question_projections" / f"{qset_hash}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing projection: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    rows = [
        ProjectionRow(
            question_id=int(r["question_id"]),
            gt_bsard_ids=[int(b) for b in r.get("gt_bsard_ids", [])],
            gt_in_pdf=[int(b) for b in r.get("gt_in_pdf", [])],
            gt_missing_from_pdf=[int(b) for b in r.get("gt_missing_from_pdf", [])],
            recall_ceiling=float(r.get("recall_ceiling", 0.0)),
        )
        for r in payload.get("rows", [])
    ]
    return payload, rows


def _all_pdf_bsard_ids(doc_id: str, cache_root: Path) -> set[int]:
    """All bsard_ids ever BSARD-linked to this PDF (drift-blind).

    Read from drift_status.json — every verdict row corresponds to a
    BSARD-linked bsard_id whose verification was attempted, regardless of
    drift outcome. This is the right "linked" set for the n_q_link counter.
    """
    p = cache_root / doc_id / "drift_status.json"
    if not p.exists():
        raise FileNotFoundError(f"missing drift_status: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    return {int(v["bsard_id"]) for v in payload.get("verdicts", [])}


def _pdf_metadata(
    doc_id: str, cache_root: Path, config_hash: str,
) -> dict[str, int]:
    """PDF-level scale metadata: pages, total chunks, BSARD-linked vs IN_PDF
    article counts."""
    pdf_meta = json.loads(
        (cache_root / doc_id / "pdf_meta.json").read_text(encoding="utf-8")
    )
    chunks = json.loads(
        (cache_root / doc_id / "configs" / config_hash / "chunks.json").read_text(encoding="utf-8")
    )
    drift = json.loads(
        (cache_root / doc_id / "drift_status.json").read_text(encoding="utf-8")
    )
    return {
        "n_pages": int(pdf_meta["n_pages"]),
        "n_chunks": int(chunks["n_chunks"]),
        "n_articles_total": int(drift["summary"]["n_bsard_articles"]),
        "n_articles_in_pdf": int(drift["summary"]["n_in_pdf"]),
    }


def _load_pool(
    doc_id: str, qset_hash: str, cache_root: Path, config_hash: str,
) -> dict[int, list[dict]]:
    p = (
        cache_root / doc_id / "results" / config_hash
        / "retrieval_pool" / f"{qset_hash}.jsonl"
    )
    if not p.exists():
        raise FileNotFoundError(f"missing pool: {p}")
    pool: dict[int, list[dict]] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pool[int(rec["question_id"])] = rec["candidates"]
    return pool


def _dense_run_for_question(candidates: list[dict]) -> dict[str, float]:
    """{bsard_id_str: max_dense_score} aggregated across chunks.

    The retrieval pool is the union of the dense (FAISS) and sparse (BM25)
    top-K's; a candidate retrieved only by the sparse side carries
    ``dense_score: null``. For dense-only analysis those are excluded — they
    were not in the dense ranking at all. Equivalently: only candidates with
    a non-null ``dense_rank`` are considered.
    """
    bsard_scores: dict[str, float] = {}
    for cand in candidates:
        ds = cand.get("dense_score")
        if ds is None:
            continue
        score = float(ds)
        if math.isnan(score):
            continue
        for bid in cand.get("bsard_ids", []):
            key = str(bid)
            if key not in bsard_scores or score > bsard_scores[key]:
                bsard_scores[key] = score
    return bsard_scores


def _ranked(bsard_scores: dict[str, float]) -> list[str]:
    return sorted(bsard_scores, key=lambda b: -bsard_scores[b])


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


# ── Per-PDF computation ──────────────────────────────────────────────────────


def _load_combined(
    doc_id: str, cache_root: Path, config_hash: str,
) -> tuple[list[ProjectionRow], dict[int, list[dict]], int]:
    """Load and merge the BSARD train + test projections + pools for one PDF.

    Returns (combined_projection_rows, combined_pool, n_questions_total).
    Asserts the two splits have disjoint question_ids — a basic sanity check
    that the BSARD split is non-overlapping as documented.
    """
    rows_combined: list[ProjectionRow] = []
    pool_combined: dict[int, list[dict]] = {}
    seen_qids: set[int] = set()
    for qset_label, qset_hash in QSET_HASHES.items():
        payload, rows = _load_projection(doc_id, qset_hash, cache_root)
        if payload.get("doc_id") != doc_id:
            raise RuntimeError(
                f"projection doc_id mismatch for {doc_id} ({qset_label}): "
                f"payload.doc_id={payload.get('doc_id')!r}"
            )
        pool = _load_pool(doc_id, qset_hash, cache_root, config_hash)
        for r in rows:
            if r.question_id in seen_qids:
                raise RuntimeError(
                    f"{doc_id}: question_id {r.question_id} appears in both "
                    f"BSARD splits — split overlap is unexpected"
                )
            seen_qids.add(r.question_id)
            rows_combined.append(r)
        pool_combined.update(pool)
    return rows_combined, pool_combined, len(rows_combined)


def _per_pdf_metrics(
    doc_id: str,
    cache_root: Path,
    config_hash: str,
) -> dict:
    rows, pool, _ = _load_combined(doc_id, cache_root, config_hash)
    pdf_linked = _all_pdf_bsard_ids(doc_id, cache_root)
    meta = _pdf_metadata(doc_id, cache_root, config_hash)

    n_q_link = 0
    n_q_evaluable = 0

    t1: dict[str, list[float]] = {
        "T1/R@10": [], "T1/R@100": [],
        "T1/MRR@10": [], "T1/NDCG@10": [],
    }
    e_: dict[str, list[float]] = {
        "E/R@10": [], "E/R@100": [],
        "E/MRR@10": [], "E/NDCG@10": [],
    }

    rc_all: list[float] = []
    rc_linked: list[float] = []
    rc_evaluable: list[float] = []

    for row in rows:
        rc_all.append(row.recall_ceiling)
        full_gt = set(row.gt_bsard_ids)
        if not (full_gt & pdf_linked):
            continue  # not linked to this PDF
        n_q_link += 1
        rc_linked.append(row.recall_ceiling)

        cands = pool.get(row.question_id)
        if cands is None:
            raise RuntimeError(
                f"{doc_id}: pool missing q={row.question_id}"
            )

        scores = _dense_run_for_question(cands)
        ranked = _ranked(scores)
        full_gt_str = {str(b) for b in full_gt}

        t1["T1/R@10"].append(recall_at_k(ranked, full_gt_str, 10))
        t1["T1/R@100"].append(recall_at_k(ranked, full_gt_str, 100))
        t1["T1/MRR@10"].append(mrr_at_k(ranked, full_gt_str, 10))
        t1["T1/NDCG@10"].append(ndcg_at_k(ranked, full_gt_str, 10))

        if row.gt_in_pdf:
            n_q_evaluable += 1
            rc_evaluable.append(row.recall_ceiling)
            gt_in_str = {str(b) for b in row.gt_in_pdf}
            e_["E/R@10"].append(recall_at_k(ranked, gt_in_str, 10))
            e_["E/R@100"].append(recall_at_k(ranked, gt_in_str, 100))
            e_["E/MRR@10"].append(mrr_at_k(ranked, gt_in_str, 10))
            e_["E/NDCG@10"].append(ndcg_at_k(ranked, gt_in_str, 10))

    return {
        "doc_id": doc_id,
        "n_pages": meta["n_pages"],
        "n_chunks": meta["n_chunks"],
        "n_articles_total": meta["n_articles_total"],
        "n_articles_in_pdf": meta["n_articles_in_pdf"],
        "n_q_link": n_q_link,
        "n_q_evaluable": n_q_evaluable,
        "T1/R@10":   _mean(t1["T1/R@10"]),
        "T1/R@100":  _mean(t1["T1/R@100"]),
        "T1/MRR@10": _mean(t1["T1/MRR@10"]),
        "T1/NDCG@10":_mean(t1["T1/NDCG@10"]),
        "E/R@10":    _mean(e_["E/R@10"]),
        "E/R@100":   _mean(e_["E/R@100"]),
        "E/MRR@10":  _mean(e_["E/MRR@10"]),
        "E/NDCG@10": _mean(e_["E/NDCG@10"]),
        "recall_ceiling_mean_all_qset":  _mean(rc_all),
        "recall_ceiling_mean_linked":    _mean(rc_linked),
        "recall_ceiling_mean_evaluable": _mean(rc_evaluable),
    }


def _aggregate_row(rows: list[dict]) -> dict:
    """Mean across the per-PDF rows for metric columns; counts summed."""
    sum_keys = [
        "n_pages", "n_chunks", "n_articles_total", "n_articles_in_pdf",
        "n_q_link", "n_q_evaluable",
    ]
    out: dict = {"doc_id": "ALL_5_PDFS"}
    for k in sum_keys:
        out[k] = int(sum(r[k] for r in rows))
    skip = {"doc_id"} | set(sum_keys)
    for k in [k for k in rows[0] if k not in skip]:
        vals = [r[k] for r in rows if not (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


# ── Output formatting ────────────────────────────────────────────────────────


def _fmt(v) -> str:
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        return f"{v:.4f}"
    return str(v)


def _format_table(rows: list[dict]) -> str:
    headers = list(rows[0].keys())
    align = ["---" if h == "doc_id" else "---:" for h in headers]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r[h]) for h in headers) + " |")
    return "\n".join(out)


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})


def _write_analysis(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    per_pdf = [r for r in rows if r["doc_id"] != "ALL_5_PDFS"]
    agg = next((r for r in rows if r["doc_id"] == "ALL_5_PDFS"), None)

    L: list[str] = []
    L.append("# Dense-only retrieval — 5 curated PDFs")
    L.append("")
    L.append(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%MZ')}. "
        f"Candidates sorted by `dense_score` (FAISS only; sparse/fused ignored). "
        f"Chunk → bsard_id aggregation by max dense score per question. Layer 2 "
        f"drift filter is applied: every chunk's `bsard_ids` is restricted to "
        f"the IN_PDF set, so the dense run can only ever surface bsard_ids "
        f"whose canonical text is genuinely in the PDF."
    )
    L.append("")
    L.append(
        "Evaluation is over the **full BSARD question set** (1108 questions = "
        "the union of BSARD's train and test splits). The split is purely a "
        "BSARD benchmark convention; RQ2 does no training, so the splits are "
        "merged here to avoid implying otherwise."
    )
    L.append("")
    L.append(
        "**Per-question identity (for evaluable questions):** "
        "`T1/R = E/R × recall_ceiling`. Layer 2 mechanically caps T1 at the "
        "ceiling — drift is the wedge between the two metrics."
    )
    L.append("")

    if agg is None or not per_pdf:
        path.write_text("\n".join(L), encoding="utf-8")
        return

    total_link = sum(r["n_q_link"] for r in per_pdf)
    total_eval = sum(r["n_q_evaluable"] for r in per_pdf)
    drifted = total_link - total_eval

    L.append("## Scope")
    L.append("")
    L.append(
        f"- Corpus: **{agg['n_pages']} pages**, **{agg['n_chunks']} chunks** "
        f"(sliding window 512 tokens, stride 256, e5-large-instruct)."
    )
    L.append(
        f"- Articles: **{agg['n_articles_total']} BSARD-linked**, of which "
        f"**{agg['n_articles_in_pdf']} are IN_PDF** post-drift "
        f"(canonical text genuinely present in the canonical PDF)."
    )
    L.append("")

    L.append("## Headline")
    L.append("")
    L.append(
        f"- Mean across the 5 PDFs: **T1/R@10 = {agg['T1/R@10']:.3f}**, "
        f"**E/R@10 = {agg['E/R@10']:.3f}**."
    )
    L.append(
        f"- Drift cost: of {total_link} questions linked to one of the 5 PDFs, "
        f"**{drifted}** are not-evaluable on their PDF (all GT drifted out)."
    )
    L.append(
        f"- Mean recall ceiling among linked questions across the 5: "
        f"{agg['recall_ceiling_mean_linked']:.3f} — this is the per-PDF "
        f"upper bound for T1 and explains the wedge to E."
    )
    L.append("")

    best_e = max(per_pdf, key=lambda r: r["E/R@10"])
    worst_e = min(per_pdf, key=lambda r: r["E/R@10"])
    gap = max(per_pdf, key=lambda r: r["E/R@10"] - r["T1/R@10"])
    L.append(
        f"- Best PDF on E/R@10: `{best_e['doc_id']}` "
        f"({best_e['E/R@10']:.3f}, n_evaluable={best_e['n_q_evaluable']})."
    )
    L.append(
        f"- Worst PDF on E/R@10: `{worst_e['doc_id']}` "
        f"({worst_e['E/R@10']:.3f}, n_evaluable={worst_e['n_q_evaluable']})."
    )
    L.append(
        f"- Biggest E↔T1 gap: `{gap['doc_id']}` "
        f"(E/R@10={gap['E/R@10']:.3f}, T1/R@10={gap['T1/R@10']:.3f}, "
        f"Δ={gap['E/R@10']-gap['T1/R@10']:.3f}; "
        f"linked-ceiling={gap['recall_ceiling_mean_linked']:.3f})."
    )
    L.append("")

    L.append("## Per-PDF breakdown")
    L.append("")
    L.append(
        "| doc_id | pages | chunks | articles | in_pdf | n_link | n_evaluable | drifted | T1/R@10 | E/R@10 | ceiling_linked |"
    )
    L.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for r in per_pdf:
        L.append(
            f"| `{r['doc_id']}` | {r['n_pages']} | {r['n_chunks']} | "
            f"{r['n_articles_total']} | {r['n_articles_in_pdf']} | "
            f"{r['n_q_link']} | {r['n_q_evaluable']} | "
            f"{r['n_q_link'] - r['n_q_evaluable']} | "
            f"{r['T1/R@10']:.3f} | {r['E/R@10']:.3f} | "
            f"{r['recall_ceiling_mean_linked']:.3f} |"
        )
    L.append("")

    L.append("## Soundness note")
    L.append("")
    L.append(
        "Under Layer 2, every chunk's `bsard_ids` is a subset of IN_PDF, so "
        "any retrieved bsard_id is also in IN_PDF. For an evaluable question, "
        "`hit ∩ full_GT = hit ∩ gt_in_pdf`, hence the per-question identity "
        "`T1/R = E/R × recall_ceiling`. There can therefore be no question "
        "where E/R = 0 but T1/R > 0 — the metric is internally consistent by "
        "construction (no need for a runtime check)."
    )
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


# ── Sanity checks ────────────────────────────────────────────────────────────


def _sanity_checks(cache_root: Path) -> None:
    rng = random.Random(0)
    for doc_id in PDFS:
        cfg = _config_hash_for(doc_id, cache_root)
        spans_p = cache_root / doc_id / "article_spans.json"
        spans_payload = json.loads(spans_p.read_text(encoding="utf-8"))
        in_pdf_ids = {int(s["bsard_id"]) for s in spans_payload.get("spans", [])}
        if not in_pdf_ids:
            raise RuntimeError(f"{doc_id}: article_spans.json has zero spans")

        for qset_label, qset_hash in QSET_HASHES.items():
            payload, rows = _load_projection(doc_id, qset_hash, cache_root)
            pool = _load_pool(doc_id, qset_hash, cache_root, cfg)

            if len(rows) != int(payload.get("n_questions", -1)):
                raise RuntimeError(
                    f"{doc_id}/{qset_label}: projection rows={len(rows)} vs "
                    f"n_questions={payload.get('n_questions')}"
                )
            if len(pool) != len(rows):
                raise RuntimeError(
                    f"{doc_id}/{qset_label}: pool size={len(pool)} vs "
                    f"projection rows={len(rows)}"
                )

            evaluable = [r for r in rows if r.gt_in_pdf]
            if evaluable:
                sample = rng.choice(evaluable)
                if not set(sample.gt_in_pdf).issubset(in_pdf_ids):
                    raise RuntimeError(
                        f"{doc_id}/{qset_label}: q={sample.question_id} "
                        f"gt_in_pdf ⊄ article_spans bsard_ids — drift filter "
                        f"/ projection inconsistent."
                    )
                # The pool is hybrid (FAISS ∪ BM25). Candidates retrieved only
                # by sparse have null dense_score; that is correct, not a bug.
                # The relevant check is that *some* candidates have a finite
                # dense_score so the dense-only ranking is meaningful.
                cands = pool.get(sample.question_id, [])
                n_dense = sum(
                    1 for c in cands
                    if c.get("dense_score") is not None
                    and not (isinstance(c.get("dense_score"), float)
                             and math.isnan(c["dense_score"]))
                )
                if n_dense == 0:
                    raise RuntimeError(
                        f"{doc_id}/{qset_label}: q={sample.question_id} pool "
                        f"has zero candidates with a non-null dense_score — "
                        f"cannot derive a dense ranking."
                    )

    print("[sanity] all checks passed")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--cache-root", type=Path, default=DEFAULT_T03_CACHE_ROOT,
    )
    args = parser.parse_args()

    _sanity_checks(args.cache_root)

    per_pdf: list[dict] = []
    for doc_id in PDFS:
        cfg = _config_hash_for(doc_id, args.cache_root)
        per_pdf.append(_per_pdf_metrics(doc_id, args.cache_root, cfg))
    rows = per_pdf + [_aggregate_row(per_pdf)]

    stamp = _utc_stamp()
    csv_path = args.output_dir / f"dense_retrieval_5pdf_{stamp}.csv"
    md_path  = args.output_dir / f"dense_retrieval_5pdf_{stamp}.md"
    _write_csv(rows, csv_path)
    _write_analysis(rows, md_path)

    print()
    print(_format_table(rows))
    print()
    print(f"CSV:      {csv_path}")
    print(f"Analysis: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
