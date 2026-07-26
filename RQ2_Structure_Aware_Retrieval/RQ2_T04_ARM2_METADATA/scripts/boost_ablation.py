"""Ablation: decouple the embedding-text axis from the boost-stage axis.

Background. T04's "variants" conflate two axes:

  axis 1 (index-time):  embedding/BM25 input text
                        raw / enriched / summary / filtered / full / terms
  axis 2 (query-time):  whether boost.apply_filters_and_boosts() runs
                        — implicitly ON for {filtered, full, terms},
                        — implicitly OFF for {raw, enriched, summary}

This script runs a clean 2-axis grid by FORCING boost on/off for the same
(unit, variant) cells. Used to confirm whether `article_full` returning
R@10=0 on the dense run was because of the doc-context-header noise
(axis 1) or because of the boost stage (axis 2) — answer: axis 1; see
CHANGE_NOTES.

Two execution modes:

  Single-stem:
      python scripts/boost_ablation.py --doc-id 1967_10_10_1967101056 -v
      python scripts/boost_ablation.py --doc-id <stem> --hybrid -v
      python scripts/boost_ablation.py --doc-id <stem> --variants raw enriched -v

  All curated stems (writes analysis/boost_ablation_v4.md):
      python scripts/boost_ablation.py --all-stems -v

Default plan: node x {raw, enriched, summary, full} x {boost off, on},
sparse-only. Cached indices are reused where compatible; sparse-only configs
will be built on demand if not yet present.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

for _sibling in _REPO.glob("RQ2_T0*/src"):
    s = str(_sibling)
    if s not in sys.path:
        sys.path.insert(0, s)

from arm2_metadata.cache import default_run_label
from arm2_metadata.retriever import run_arm2_metadata
from data_loader.models import RetrievalResult
from evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from shared.embeddings import EmbeddingModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
DEFAULT_PDF_DIR = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "pdfs"

BSARD_DB_PATH = _REPO / "dataset_creation_output" / "bsard_corpus.db"
PDF_DOC_MAP_CSV = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "csv" / "pdf_document_map.csv"
AZUREDI_DIR = _REPO / "RQ2_T04_ARM2_METADATA" / "azuredi"
T04_INDEX_ROOT = _REPO / "RQ2_T04_ARM2_METADATA" / "data"
ANALYSIS_DIR = _REPO / "RQ2_T04_ARM2_METADATA" / "analysis"
GT_TEST = _REPO / "RQ2_T07_EVALUATION" / "ground_truth" / "bsard_test.json"
GT_TRAIN = _REPO / "RQ2_T07_EVALUATION" / "ground_truth" / "bsard_train.json"
OUT_MD = ANALYSIS_DIR / "boost_ablation_v4.md"

# The curated 5-stem set used by --all-stems. Mirrors the registry at
# RQ2_T00_ORCHESTRATOR/data/selected_pdfs.json.
CURATED_STEMS: list[tuple[str, int, str]] = [
    ("1967_10_10_1967101056", 5, "Code Judiciaire (smaller)"),
    ("1867_06_08_1867060850", 6, "Code Penal"),
    ("1804_03_21_1804032150", 9, "Code Civil"),
    ("2003_07_17_2013A31614", 8, "Loi (rentals)"),
    ("1967_10_10_1967101055", 7, "Code Judiciaire (larger)"),
]


def _resolve_doc(
    doc_id: str,
    pdf_dir: Path,
    pdf_doc_map_csv: Path,
) -> tuple[str, int, Path]:
    """Map a BSARD PDF stem to (pdf_filename, azure_doc_id, pdf_path)."""
    import pandas as pd
    df = pd.read_csv(pdf_doc_map_csv, encoding="utf-8")
    target = f"{doc_id}.pdf"
    row = df[df["pdf_filename"] == target]
    if row.empty:
        raise ValueError(
            f"doc_id {doc_id!r} not in {pdf_doc_map_csv} — has_azure_extraction "
            f"is false for this PDF, or the stem is wrong."
        )
    return target, int(row.iloc[0]["document_id"]), pdf_dir / target


def _build_subset(azure_doc_id: int) -> tuple[dict[str, list[int]], set[str]]:
    """Per-doc question subset (same construction as compare_t03_vs_t04.py)."""
    from arm2_metadata.azuredi_loader import load_azuredi_corpus
    from arm2_metadata.bsard_link import link_corpus_to_bsard

    nodes, _, _ = load_azuredi_corpus(AZUREDI_DIR, PDF_DOC_MAP_CSV)
    nodes = link_corpus_to_bsard(nodes, BSARD_DB_PATH)
    reachable = {b for n in nodes for b in n.bsard_ids if n.doc_id == azure_doc_id}

    gt: dict[str, list[int]] = {}
    for path in (GT_TEST, GT_TRAIN):
        for qid, bids in json.loads(path.read_text(encoding="utf-8")).items():
            inter = sorted(set(bids) & reachable)
            if inter:
                gt[str(qid)] = inter
    return gt, set(gt)


def _bsard_ranking(result: RetrievalResult) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in result.ranked_items:
        meta = item.get("metadata") or {}
        if meta.get("bsard_id") is not None:
            ids = [str(int(meta["bsard_id"]))]
        elif meta.get("bsard_ids"):
            ids = [str(int(b)) for b in meta["bsard_ids"]]
        else:
            ids = [str(item["id"])]
        for bid in ids:
            if bid not in seen:
                out.append(bid)
                seen.add(bid)
    return out


def _metrics(results: list[RetrievalResult], gt: dict[str, list[int]]) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for r in results:
        if r.query_id not in gt:
            continue
        rel = {str(b) for b in gt[r.query_id]}
        ranked = _bsard_ranking(r)
        rows.append({
            "R@10": recall_at_k(ranked, rel, 10),
            "R@100": recall_at_k(ranked, rel, 100),
            "MRR@10": mrr_at_k(ranked, rel, 10),
            "nDCG@10": ndcg_at_k(ranked, rel, 10),
        })
    if not rows:
        return {"n_queries": 0}
    n = len(rows)
    return {
        "R@10": sum(r["R@10"] for r in rows) / n,
        "R@100": sum(r["R@100"] for r in rows) / n,
        "MRR@10": sum(r["MRR@10"] for r in rows) / n,
        "nDCG@10": sum(r["nDCG@10"] for r in rows) / n,
        "n_queries": n,
    }


def _run_one_grid(
    *,
    doc_id: str,
    azure_doc_id: int,
    pdf_path: Path,
    units: list[str],
    variants: list[str],
    sparse_only: bool,
    embedding_model,
    tokenizer,
    config_base: dict,
) -> list[dict]:
    """Run the (unit x variant x boost on/off) grid for one stem.

    Returns a list of dicts: {unit, variant, boost, metrics}. Caches are
    reused where compatible; sparse-only configs are built on demand.
    """
    gt, qids = _build_subset(azure_doc_id)
    print(f"  subset: {len(gt)} questions reachable on doc_id={azure_doc_id}")

    cfg = json.loads(json.dumps(config_base))  # deep copy
    cfg["arm2"]["sparse_only"] = sparse_only

    grid_rows: list[dict] = []
    for unit in units:
        for variant in variants:
            for boost_override in (False, True):
                method = f"{unit}/{variant}/boost={'on' if boost_override else 'off'}"
                print(f"    {method} …", end=" ", flush=True)
                t0 = time.perf_counter()
                results = run_arm2_metadata(
                    azuredi_dir=AZUREDI_DIR,
                    pdf_document_map_csv=PDF_DOC_MAP_CSV,
                    bsard_db_path=BSARD_DB_PATH,
                    pdf_path=pdf_path if pdf_path.exists() else None,
                    embedding_model=embedding_model,
                    tokenizer=tokenizer,
                    llm=None,
                    index_root=T04_INDEX_ROOT,
                    config=cfg,
                    doc_id=doc_id,
                    azure_doc_id=azure_doc_id,
                    variant=variant,
                    unit=unit,
                    run_label=f"ablate_boost_{default_run_label()}",
                    question_ids=qids,
                    split=None,
                    apply_boost_override=boost_override,
                )
                m = _metrics(results, gt)
                elapsed = time.perf_counter() - t0
                print(f"R@10={m.get('R@10', 0):.4f}  MRR@10={m.get('MRR@10', 0):.4f}  ({elapsed:.1f}s)")
                grid_rows.append({
                    "unit": unit, "variant": variant,
                    "boost": "on" if boost_override else "off",
                    **m,
                })
    return grid_rows


def _format_stem_table(rows: list[dict], stem: str, doc_id: int, title: str) -> str:
    """Markdown table for one stem's grid + the boost-effect deltas."""
    lines: list[str] = []
    lines.append(f"### {stem}  ·  doc_id {doc_id}  ·  {title}")
    lines.append("")
    lines.append("| unit | variant | boost | R@10 | R@100 | MRR@10 | nDCG@10 | n_q |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['unit']} | {r['variant']} | {r['boost']} | "
            f"{r.get('R@10', 0):.4f} | {r.get('R@100', 0):.4f} | "
            f"{r.get('MRR@10', 0):.4f} | {r.get('nDCG@10', 0):.4f} | "
            f"{int(r.get('n_queries', 0))} |"
        )
    # Per-(unit, variant) boost effect: on - off.
    by_key: dict[tuple[str, str, str], dict] = {(r["unit"], r["variant"], r["boost"]): r for r in rows}
    keys = sorted({(r["unit"], r["variant"]) for r in rows})
    lines.append("")
    lines.append("**Boost effect (on − off):**")
    lines.append("")
    lines.append("| unit | variant | ΔR@10 | ΔR@100 | ΔMRR@10 | ΔnDCG@10 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for unit, variant in keys:
        off = by_key.get((unit, variant, "off"), {})
        on = by_key.get((unit, variant, "on"), {})
        if not off or not on:
            continue
        lines.append(
            f"| {unit} | {variant} | {on.get('R@10', 0) - off.get('R@10', 0):+.4f} | "
            f"{on.get('R@100', 0) - off.get('R@100', 0):+.4f} | "
            f"{on.get('MRR@10', 0) - off.get('MRR@10', 0):+.4f} | "
            f"{on.get('nDCG@10', 0) - off.get('nDCG@10', 0):+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_aggregate_md(all_rows: dict[str, list[dict]], mode_label: str, units: list[str], variants: list[str]) -> None:
    """Write analysis/boost_ablation_v4.md."""
    lines: list[str] = []
    lines.append("# T04 boost ablation — v4 5-stem grid")
    lines.append("")
    lines.append(f"Mode: **{mode_label}**.")
    lines.append("")
    lines.append(
        "This file separates T04's two design axes — embedding-text variant and "
        "query-time boost stage — by running each (unit, variant) cell twice "
        "(once with boost on, once with boost off) on each of the curated 5 stems. "
        "Generated by `RQ2_T04_ARM2_METADATA/scripts/boost_ablation.py --all-stems`."
    )
    lines.append("")
    lines.append(
        "Caveat under sparse-only on a single-PDF corpus: the v4 enricher drops "
        "the constant `[Document: ...]` header on single-doc corpora, so node "
        "`raw / enriched / full` collapse to identical BM25 input text. Their "
        "rows will therefore be bit-identical; the only embedding-text "
        "differentiator is `summary` (and on `article` units, `raw`/`full` "
        "similarly collapse). Run with `--hybrid` to differentiate via dense "
        "embeddings."
    )
    lines.append("")
    lines.append("## Per-stem grid")
    lines.append("")
    for stem, doc_id, title in CURATED_STEMS:
        rows = all_rows.get(stem, [])
        if not rows:
            lines.append(f"### {stem} — *no rows (run failed or skipped)*")
            lines.append("")
            continue
        lines.append(_format_stem_table(rows, stem, doc_id, title))

    # Cross-stem aggregate boost effect
    lines.append("## Cross-stem aggregate boost effect (mean Δ on − off)")
    lines.append("")
    lines.append("| unit | variant | ΔR@10 | ΔR@100 | ΔMRR@10 | ΔnDCG@10 | n_stems |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for unit in units:
        for variant in variants:
            ds_r10, ds_r100, ds_mrr, ds_ndcg, n = [], [], [], [], 0
            for stem, _, _ in CURATED_STEMS:
                rs = all_rows.get(stem, [])
                off = next((r for r in rs if r["unit"] == unit and r["variant"] == variant and r["boost"] == "off"), None)
                on = next((r for r in rs if r["unit"] == unit and r["variant"] == variant and r["boost"] == "on"), None)
                if not off or not on:
                    continue
                ds_r10.append(on.get("R@10", 0) - off.get("R@10", 0))
                ds_r100.append(on.get("R@100", 0) - off.get("R@100", 0))
                ds_mrr.append(on.get("MRR@10", 0) - off.get("MRR@10", 0))
                ds_ndcg.append(on.get("nDCG@10", 0) - off.get("nDCG@10", 0))
                n += 1
            if not n:
                continue
            mean = lambda xs: sum(xs) / len(xs)
            lines.append(
                f"| {unit} | {variant} | {mean(ds_r10):+.4f} | {mean(ds_r100):+.4f} | "
                f"{mean(ds_mrr):+.4f} | {mean(ds_ndcg):+.4f} | {n} |"
            )
    lines.append("")
    lines.append("## Reading the table")
    lines.append("")
    lines.append("- **Positive Δ** → the boost stage helped (turning it on improved the metric).")
    lines.append("- **Zero Δ** → boost stage was a no-op for that cell (typically because no boost signals fired for any query — e.g. no term matches, no jurisdiction signal, no article references).")
    lines.append("- **Negative Δ** → boost actively hurt (the multipliers pulled non-GT items above GT items). Look at the boost config in `arm2_metadata/boost.py`: `term_match=1.3`, `used_in=1.5`, `jurisdiction=1.1`, `drop_non_effective=True`.")
    lines.append("")
    lines.append(
        "The big question this file answers: **does the boost stage do meaningful work, "
        "or is index-time enrichment the whole signal?** Compare the magnitude of "
        "the Δ values here against the variant-to-variant Δ in "
        "`comparison_summary_v4.md` (e.g. `node_summary` vs `node_raw` on doc 9, "
        "Δ R@10 = +0.040). If boost-on/off Δ is small relative to variant-to-variant Δ, "
        "the enrichment carries the signal and boost is mostly cosmetic."
    )
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_MD}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--doc-id", type=str, default=None,
        help="BSARD PDF stem. Required unless --all-stems is set.",
    )
    parser.add_argument(
        "--all-stems", action="store_true",
        help="Loop the curated 5 stems and write analysis/boost_ablation_v4.md.",
    )
    parser.add_argument(
        "--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
        help="Source PDF directory (default: %(default)s).",
    )
    parser.add_argument("--variants", nargs="+",
                        default=["raw", "enriched", "summary", "full"],
                        help="Embedding variants to ablate (default: raw enriched summary full).")
    parser.add_argument("--units", nargs="+", default=["node"],
                        choices=["article", "node"],
                        help="Units to ablate (default: node).")
    parser.add_argument("--hybrid", action="store_true",
                        help="Run dense+sparse (slow). Default sparse-only.")
    parser.add_argument(
        "--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
        help="Sentence-transformer name (only loaded under --hybrid).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.all_stems and not args.doc_id:
        parser.error("either --doc-id or --all-stems is required")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("arm2_metadata").setLevel(logging.INFO)

    sparse_only = not args.hybrid
    mode_label = "sparse-only (BM25 only)" if sparse_only else "hybrid (dense + sparse + RRF)"
    print(f"Mode: {mode_label}")
    print(f"Grid: units={args.units}  variants={args.variants}  boost=off|on")

    print("Loading models …", flush=True)
    embedding_model = None if sparse_only else EmbeddingModel(model_name=args.embedding_model)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)

    config_base = {
        "models": {"embedding_model": args.embedding_model, "tokenizer": args.embedding_model},
        "arm2": {
            "retrieval_top_k": 100, "top_k": 100, "max_tokens": 512,
            "use_llm_classifier": False, "sparse_only": sparse_only,
            "boost": {
                "term_match": 1.3, "used_in": 1.5, "jurisdiction": 1.1,
                "drop_non_effective": True, "hard_filter_jurisdiction": False,
            },
        },
    }

    if args.all_stems:
        all_rows: dict[str, list[dict]] = {}
        for stem, doc_id, title in CURATED_STEMS:
            print(f"\n=== {stem}  (doc_id {doc_id}, {title}) ===")
            _, azure_doc_id, default_pdf_path = _resolve_doc(stem, args.pdf_dir, PDF_DOC_MAP_CSV)
            try:
                rows = _run_one_grid(
                    doc_id=stem, azure_doc_id=azure_doc_id, pdf_path=default_pdf_path,
                    units=args.units, variants=args.variants, sparse_only=sparse_only,
                    embedding_model=embedding_model, tokenizer=tokenizer,
                    config_base=config_base,
                )
                all_rows[stem] = rows
            except Exception as e:
                logger.exception("stem %s failed: %s", stem, e)
                all_rows[stem] = []
        _write_aggregate_md(all_rows, mode_label, args.units, args.variants)
        return 0

    # Single-stem path.
    _, azure_doc_id, default_pdf_path = _resolve_doc(args.doc_id, args.pdf_dir, PDF_DOC_MAP_CSV)
    print(f"\n=== {args.doc_id}  (doc_id {azure_doc_id}) ===")
    rows = _run_one_grid(
        doc_id=args.doc_id, azure_doc_id=azure_doc_id, pdf_path=default_pdf_path,
        units=args.units, variants=args.variants, sparse_only=sparse_only,
        embedding_model=embedding_model, tokenizer=tokenizer,
        config_base=config_base,
    )

    print("\n" + "=" * 78)
    print(f"  {'method':<32s}  {'R@10':>6s}  {'MRR@10':>7s}  {'nDCG@10':>8s}")
    print("-" * 78)
    for r in rows:
        method = f"{r['unit']}/{r['variant']}/boost={r['boost']}"
        print(f"  {method:<32s}  {r.get('R@10', 0):>6.3f}  {r.get('MRR@10', 0):>7.3f}  {r.get('nDCG@10', 0):>8.3f}")

    print("\nBoost effect (R@10 boost_on - boost_off):")
    by_key = {(r["unit"], r["variant"], r["boost"]): r for r in rows}
    for unit in args.units:
        for variant in args.variants:
            off = by_key.get((unit, variant, "off"))
            on = by_key.get((unit, variant, "on"))
            if off and on:
                d = on.get("R@10", 0) - off.get("R@10", 0)
                print(f"  {unit}/{variant}: off={off['R@10']:.3f}  on={on['R@10']:.3f}  delta={d:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
