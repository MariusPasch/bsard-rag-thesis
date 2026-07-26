"""Load each arm's *persisted* per-query rankings + the per-PDF ground truth.

No retrieval is ever run here — every arm already wrote its ranked results to
disk, and evaluation is a pure function of (ranked bsard_id lists, GT). Each
loader returns ``list[RetrievalResult]`` shaped exactly as T07's
``evaluation.adapter`` expects: items carry ``id``, ``score`` and a ``metadata``
dict holding ``bsard_ids`` (chunks, T03) or ``bsard_id`` (articles, T04/T05).

Raw on-disk formats (all under the shared data root):
  T03  pdf_cache/<stem>/results/<hash>/retrieval_pool/<qset>.jsonl
         row: {question_id, question_text, candidates:[{rank, chunk_id,
               fused_score, bsard_ids:[...], ...}]}              (top-200 pool)
  T04  RQ2_T04.../<stem>/results/<hash>/compare_<ts>.jsonl
         row: {query_id, query_text, ranked_items:[{id, score,
               metadata:{bsard_ids|bsard_id, ...}}]}             (top-100, per variant)
         variant identity from ../configs/<hash>/manifest.json["chunking"]
  T05  RQ2_T05.../<stem>/results/q<qid>.json
         {query_id, query_text, ranked_items:[{id, score, text,
               metadata:{bsard_id, ...}}], method, cost, trace}  (pad_to_k=100)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import paths

# Use the canonical RetrievalResult when the sibling package is installed;
# fall back to a structurally-identical shim so loaders (and their smoke test)
# work even before `pip install -e ../RQ2_T02_DATA_LOADER`. The shim exposes the
# three attributes evaluation.adapter touches: query_id, ranked_items, cost.
try:  # pragma: no cover - import-environment dependent
    from data_loader.models import RetrievalResult
except Exception:  # noqa: BLE001
    from dataclasses import dataclass, field

    @dataclass
    class RetrievalResult:  # type: ignore[no-redef]
        query_id: str
        query_text: str
        ranked_items: list[dict]
        method: str
        cost: dict = field(default_factory=dict)
        trace: dict | None = None


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(stem: str) -> dict[str, list[int]]:
    """Per-PDF curated GT: {query_id_str: [bsard_id_int, ...]}.

    These ``runs/t04_<stem>.json`` files are the per-PDF question subsets every
    arm scored against (key counts match each PDF's question set exactly), so
    they are the single consistent GT for cross-arm comparison.
    """
    raw = json.loads(paths.gt_path(stem).read_text(encoding="utf-8"))
    return {str(qid): [int(b) for b in bids] for qid, bids in raw.items()}


# ---------------------------------------------------------------------------
# T03 — ARM1 naive (chunk pools, top-200 RRF fused)
# ---------------------------------------------------------------------------

def load_t03(stem: str) -> list[RetrievalResult]:
    """Load T03's fused retrieval pool for ``stem`` as RetrievalResults.

    Items are chunks; ``metadata.bsard_ids`` lets T07 aggregate to article level
    by max chunk score (the designed Arm-1 path). Returns [] if absent.
    """
    root = paths.T03_PDF_CACHE / stem / "results"
    if not root.exists():
        return []
    out: list[RetrievalResult] = []
    for hash_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        pool_dir = hash_dir / "retrieval_pool"
        if not pool_dir.exists():
            continue
        for fp in sorted(pool_dir.glob("*.jsonl")):
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    items = [
                        {
                            "id": c.get("chunk_id"),
                            "score": float(c.get("fused_score", 0.0)),
                            "metadata": {
                                "bsard_ids": [int(b) for b in (c.get("bsard_ids") or [])],
                            },
                        }
                        for c in d.get("candidates", [])
                    ]
                    out.append(
                        RetrievalResult(
                            query_id=str(d["question_id"]),
                            query_text=d.get("question_text", ""),
                            ranked_items=items,
                            method="T03_naive",
                            cost={},
                        )
                    )
        if out:  # one pool dir is enough; first config hash present
            break
    return out


# ---------------------------------------------------------------------------
# T04 — ARM2 metadata (article/node level, many variants)
# ---------------------------------------------------------------------------

def _t04_label(chunking: dict) -> str:
    fusion = "sparse" if chunking.get("sparse_only") else "hybrid"
    return f"T04_{chunking.get('variant')}_{chunking.get('unit')}_{fusion}"


def load_t04_variants(
    stem: str, *, include_ablations: bool = False
) -> dict[str, list[RetrievalResult]]:
    """Load every persisted T04 variant for ``stem``.

    Returns {variant_label: list[RetrievalResult]}. When a (variant, unit,
    fusion) appears under several config hashes, keep the one with the highest
    ``linker_version`` then the most recent file (the v4 sweep wins over v3).
    Boost-ablation runs are skipped unless ``include_ablations`` is set.
    """
    base = paths.T04_BASE / stem
    results_root = base / "results"
    if not results_root.exists():
        return {}

    # label → (linker_version, mtime, RetrievalResult list)
    best: dict[str, tuple[int, float, list[RetrievalResult]]] = {}

    for hash_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        prefix = "ablate_boost_" if include_ablations else "compare_"
        jsonls = sorted(hash_dir.glob(f"{prefix}*.jsonl"))
        if not include_ablations:
            jsonls = sorted(hash_dir.glob("compare_*.jsonl"))
        if not jsonls:
            continue
        manifest_fp = base / "configs" / hash_dir.name / "manifest.json"
        if not manifest_fp.exists():
            continue
        chunking = json.loads(manifest_fp.read_text(encoding="utf-8")).get("chunking", {})
        label = _t04_label(chunking)
        linker = int(chunking.get("linker_version", 0))

        fp = jsonls[-1]
        mtime = fp.stat().st_mtime
        prev = best.get(label)
        if prev is not None and (linker, mtime) <= (prev[0], prev[1]):
            continue

        res = _read_t04_jsonl(fp)
        best[label] = (linker, mtime, res)

    return {label: triple[2] for label, triple in sorted(best.items())}


def _read_t04_jsonl(fp: Path) -> list[RetrievalResult]:
    out: list[RetrievalResult] = []
    with fp.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            items = []
            for it in d.get("ranked_items", []):
                md = it.get("metadata") or {}
                # Normalise to what the adapter resolves (bsard_ids | bsard_id).
                items.append(
                    {
                        "id": it.get("id"),
                        "score": float(it.get("score", 0.0)),
                        "metadata": md,
                    }
                )
            out.append(
                RetrievalResult(
                    query_id=str(d["query_id"]),
                    query_text=d.get("query_text", ""),
                    ranked_items=items,
                    method="T04",
                    cost=d.get("cost", {}) or {},
                )
            )
    return out


# ---------------------------------------------------------------------------
# T05 — ARM2 PageIndex (article level, per-query JSON, rich cost)
# ---------------------------------------------------------------------------

def load_t05(stem: str, *, snapshot: str = "results") -> list[RetrievalResult]:
    """Load T05 PageIndex rankings for ``stem`` from ``snapshot`` dir.

    ``snapshot`` defaults to the live ``results/`` (post-fix + chapter-then-law
    padding). Pass a ``results_*`` snapshot name to score an earlier state.
    """
    d_dir = paths.T05_BASE / stem / snapshot
    if not d_dir.exists():
        return []
    out: list[RetrievalResult] = []
    for fp in sorted(d_dir.glob("q*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        items = []
        for it in d.get("ranked_items", []):
            md = it.get("metadata") or {}
            items.append(
                {
                    "id": it.get("id"),
                    "score": float(it.get("score", 0.0)),
                    "metadata": {"bsard_id": md.get("bsard_id")},
                }
            )
        out.append(
            RetrievalResult(
                query_id=str(d["query_id"]),
                query_text=d.get("query_text", ""),
                ranked_items=items,
                method=d.get("method", "T05_pageindex"),
                cost=d.get("cost", {}) or {},
            )
        )
    return out


# ---------------------------------------------------------------------------
# T03 chunk→bsard overlap weights (for opt-in W/ weighted metrics)
# ---------------------------------------------------------------------------

def load_t03_weights(stem: str) -> dict[tuple[str, int], float] | None:
    """Load T07's cached chunk→article overlap weights for T03 chunks.

    Returns {(chunk_id, bsard_id): weight} or None if the cache is absent.
    These are precomputed by T07 (weight = fraction of the article's tokens the
    chunk covers, capped at 1.0); we only read them.
    """
    fp = paths.t03_weights_csv(stem)
    if fp is None:
        return None
    out: dict[tuple[str, int], float] = {}
    with fp.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[(row["chunk_id"], int(row["bsard_id"]))] = float(row["weight"])
    return out


# ---------------------------------------------------------------------------
# Assemble all arms for one stem
# ---------------------------------------------------------------------------

def load_all_arms(
    stem: str, *, t04_include_ablations: bool = False
) -> dict[str, list[RetrievalResult]]:
    """Build the {method: list[RetrievalResult]} mapping for one PDF stem.

    Method-naming note: none of the keys contain the substring ``arm1``, so
    T07's ``is_chunk_based`` stays False and the weighted-metric / weights-cache
    machinery (which needs the BSARD db) is never triggered. Binary T1/T2
    metrics are still correct — the adapter resolves chunk ``bsard_ids`` from
    metadata and aggregates to article level by max score regardless.
    """
    arms: dict[str, list[RetrievalResult]] = {}
    t03 = load_t03(stem)
    if t03:
        arms["T03_naive"] = t03
    arms.update(load_t04_variants(stem, include_ablations=t04_include_ablations))
    t05 = load_t05(stem)
    if t05:
        arms["T05_pageindex"] = t05
    return arms
