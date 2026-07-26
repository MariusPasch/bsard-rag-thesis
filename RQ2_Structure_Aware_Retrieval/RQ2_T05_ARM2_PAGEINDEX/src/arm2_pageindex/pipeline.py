"""Batch runner for Arm 2B over a list of queries.

Designed to be called from both:
  * the Azure GPU notebook (real Ollama, full subset run);
  * a local notebook or smoke script (stub LLM, validation).

Persistence layout (when ``out_dir`` is given):

    <out_dir>/
    ├── manifest.json         # run-level summary
    └── q<query_id>.json      # one RetrievalResult per query (asdict-style)

A re-run with the same ``out_dir`` is **idempotent**: any ``q<qid>.json``
already on disk is reloaded instead of re-issuing LLM calls. Delete
the file (or the whole directory) to force a fresh run.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from data_loader.models import RetrievalResult

from .navigator import NavigatorConfig
from .retriever import DEFAULT_METHOD_NAME, run_arm2b
from .tree_builder import ToCNode

if TYPE_CHECKING:
    from shared.llm import LLMClient

logger = logging.getLogger(__name__)


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _result_from_dict(d: dict) -> RetrievalResult:
    return RetrievalResult(
        query_id=d["query_id"],
        query_text=d["query_text"],
        ranked_items=d.get("ranked_items", []),
        method=d.get("method", DEFAULT_METHOD_NAME),
        cost=d.get("cost", {}),
        trace=d.get("trace"),
    )


def _dump_result(result: RetrievalResult, target: Path) -> None:
    target.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_subset(
    *,
    tree: ToCNode,
    queries: list[dict],
    llm: "LLMClient",
    cfg: NavigatorConfig | None = None,
    out_dir: Path | str | None = None,
    method_name: str = DEFAULT_METHOD_NAME,
    progress: bool = True,
    log_every: int = 10,
    on_query_done: Callable[[int, RetrievalResult], None] | None = None,
) -> list[RetrievalResult]:
    """Run Arm 2B over a list of queries.

    Args:
        tree:           Pre-built tree. Reused across all queries.
        queries:        ``[{"query_id": str, "query_text": str}, ...]``.
        llm:            LLMClient (real or stub).
        cfg:            Navigator config; ``None`` -> defaults.
        out_dir:        Directory to write per-query JSON + manifest. ``None``
                        keeps results in memory only (useful for tests).
        method_name:    Stamped on every RetrievalResult.
        progress:       If True and ``tqdm`` is importable, show a bar.
        log_every:      Emit a running stats line every N queries.
        on_query_done:  Optional hook called with (index, result) per query
                        for live notebook stats.
    """
    cfg = cfg or NavigatorConfig()
    out_dir_p = Path(out_dir) if out_dir is not None else None
    if out_dir_p is not None:
        out_dir_p.mkdir(parents=True, exist_ok=True)

    iterator: Iterable = enumerate(queries)
    if progress:
        try:
            from tqdm import tqdm
            iterator = enumerate(tqdm(queries, desc="T05"))
        except ImportError:
            pass

    results: list[RetrievalResult] = []
    t0 = time.perf_counter()
    cumulative = {
        "llm_calls": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": 0.0,
        "parse_failures": 0,
        "n_cached_skip": 0,
    }
    exit_reasons: dict[str, int] = {}

    for i, q in iterator:
        qid = str(q["query_id"])
        target = out_dir_p / f"q{qid}.json" if out_dir_p else None

        if target is not None and target.exists():
            try:
                cached = json.loads(target.read_text(encoding="utf-8"))
                result = _result_from_dict(cached)
                cumulative["n_cached_skip"] += 1
                cached_exit = (result.cost or {}).get("exit_reason", "?")
                exit_reasons[cached_exit] = exit_reasons.get(cached_exit, 0) + 1
                results.append(result)
                if on_query_done:
                    on_query_done(i, result)
                continue
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Discarding bad cache file %s (%s); re-running query",
                    target, exc,
                )

        result = run_arm2b(
            query=q["query_text"],
            query_id=qid,
            tree=tree,
            llm=llm,
            cfg=cfg,
            method_name=method_name,
        )
        for k in ("llm_calls", "tokens_in", "tokens_out", "latency_ms", "parse_failures"):
            cumulative[k] += result.cost.get(k, 0)
        exit_reason = result.cost.get("exit_reason", "?")
        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1

        if target is not None:
            _dump_result(result, target)
        results.append(result)
        if on_query_done:
            on_query_done(i, result)

        if (i + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            run_count = (i + 1) - cumulative["n_cached_skip"]
            avg = elapsed / max(run_count, 1)
            eta = avg * (len(queries) - i - 1)
            parse_rate = (
                cumulative["parse_failures"] / max(cumulative["llm_calls"], 1)
            )
            logger.info(
                "[Q %d/%d] elapsed=%.1fs avg=%.2fs/q ETA=%.0fs "
                "calls=%d parse_fail=%.1f%% cache_skip=%d",
                i + 1, len(queries), elapsed, avg, eta,
                cumulative["llm_calls"], 100.0 * parse_rate,
                cumulative["n_cached_skip"],
            )

    elapsed = time.perf_counter() - t0
    n_fresh = len(queries) - cumulative["n_cached_skip"]
    if out_dir_p is not None and n_fresh == 0:
        # Pure cache replay - the existing manifest already describes the state
        # of this directory; rewriting it with zeroed totals would erase
        # information about the cold run that produced these files.
        logger.info(
            "Pure cache replay (%d/%d hits) - preserving existing manifest at %s",
            cumulative["n_cached_skip"], len(queries), out_dir_p / "manifest.json",
        )
        return results

    if out_dir_p is not None:
        manifest = {
            "method": method_name,
            "n_queries": len(queries),
            "n_cached_skip": cumulative["n_cached_skip"],
            "tree_root_id": tree.node_id,
            "elapsed_s": round(elapsed, 1),
            "totals": {k: cumulative[k] for k in ("llm_calls", "tokens_in", "tokens_out", "latency_ms", "parse_failures")},
            "exit_reasons": exit_reasons,
            "saved_at": _now_utc(),
            "config": {
                "max_laws": cfg.max_laws,
                "max_chapters_per_law": cfg.max_chapters_per_law,
                "max_articles_per_chapter": cfg.max_articles_per_chapter,
                "max_iterations": cfg.max_iterations,
                "skip_law_selection_if_single_doc": cfg.skip_law_selection_if_single_doc,
                "score_threshold": cfg.score_threshold,
            },
        }
        (out_dir_p / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Wrote %d per-query JSON files + manifest to %s "
            "(elapsed %.1fs, %d LLM calls, %d cache hits)",
            len(queries), out_dir_p, elapsed,
            cumulative["llm_calls"], cumulative["n_cached_skip"],
        )

    return results


def load_subset_results(out_dir: Path | str) -> list[RetrievalResult]:
    """Load all q<qid>.json files from a previous run_subset call."""
    p = Path(out_dir)
    results: list[RetrievalResult] = []
    for path in sorted(p.glob("q*.json")):
        try:
            results.append(_result_from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Skipping malformed result %s (%s)", path, exc)
    return results
