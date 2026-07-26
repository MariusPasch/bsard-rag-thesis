"""Arm 2A (Metadata-enriched) retrieved-article deep-dive — compute engine.

Profiles the *retrieved articles* of Arm 2A (T04 metadata index over AzureDI
nodes / article units) along the properties that matter for downstream answer
generation, per PDF and aggregated over the curated 5-PDF set (725 questions).
Pure read from the persisted one-path artifacts — no retrieval, indexing or
linking is ever re-run (STYLEGUIDE §6 / EVALUATION_METHODOLOGY §4).

Variant set (confirmed with the user 2026-06-01). Three variants, the boost-inert
duplicates dropped (enriched_node/full_node ≡ raw_node, full_article ≡
raw_article — byte-identical, see reports/cross_arm_analysis.md §2):
  - ``summary_node``  — the **canonical** Arm 2A (primary).
  - ``raw_node``      — isolates the index-time enrichment lift vs summary_node.
  - ``raw_article``   — the node-vs-article granularity / coverage contrast
                        (whole-unit: coverage ≡ 1.0 on a hit, by construction).

Group A (article-level retrieval) and Group D1 (overlap with gold) are computed
by the **shared** ``deep_dive_common`` primitives — provably identical code to the
Arm-1 dive, so a later Arm-2A-vs-Arm-1 (or vs Arm-2B) synthesis is a simple join.

Two senses of *k* (labelled throughout):
  - **k = articles** — A1 recall, A2 hit, A5 rank-of-gold, D1 Jaccard: on the
    article ranking (nodes aggregated to bsard articles by *max node score*, the
    documented T04 adapter path). Asserted to reproduce the published ``T1/R@k``
    for every variant before any analysis.
  - **k = nodes** — A3 article-context precision, A4 distinct units, A6
    gold/distractor mix and all of Group B coverage: on the top-k *nodes the
    generator actually receives*. For ``raw_article`` a "node" already is an
    article, so these collapse onto the article level by design.

Group B — token coverage (node variants only). For each *retrieved* gold article
coverage@k = capped Σ over the top-k retrieved NODES of w(node, article).
**Two weights** are persisted (user decision 2026-06-01: compute both, headline
the DB-overlap):
  - ``w_db_overlap`` (PRIMARY) — the e5-token analogue of Arm 1's char-overlap
    weight: |multiset token overlap of node page_content with the BSARD-DB
    article text| / |DB article tokens|. Directly comparable to Arm 1's coverage
    (0.806 / 18% fragment / 64% near-complete). Because Arm 2A nodes have no
    shared char-coordinate system with the DB text (unlike T03's chunks), the
    mechanism is token-multiset overlap rather than char-span overlap — the
    metric still means "fraction of the DB article's text the node carries".
  - ``w_node_share`` (alongside) — tokens(node) / Σ tokens(nodes of that
    article) = "fraction of the *indexed* article carried by the node". Sums to
    1.0 per article by construction; the brief's self-contained formula.

Determinism: seed = 42 for the only stochastic step (bootstrap CIs). Idempotent.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import deep_dive_common as common
from . import loaders, paths

# --- config ----------------------------------------------------------------
K_ARTICLE = [5, 10, 20, 100]
K_NODE = [5, 10, 20]
FRAGMENT_TAU = 0.5
NEAR_COMPLETE_TAU = 0.9
SEED = common.SEED
N_BOOT = common.N_BOOT
POOL_DEPTH = 100                       # T04 persists the top-100 (vs Arm 1's 200)
TOKENIZER_NAME = "intfloat/multilingual-e5-large-instruct"

PRIMARY = "T04_summary_node_hybrid"
VARIANTS = [
    "T04_summary_node_hybrid",
    "T04_raw_node_hybrid",
    "T04_raw_article_hybrid",
]
NODE_VARIANTS = ["T04_summary_node_hybrid", "T04_raw_node_hybrid"]  # Group B applies
ARTICLE_VARIANTS = ["T04_raw_article_hybrid"]                       # whole-unit
# Raw FR page_content + node→bsard linking come from the raw_node index (its
# ``text`` field IS the page_content). summary_node shares the same node set, so
# its retrieved nodes carry the same FR text — coverage is defined on that text
# for both, exactly as the brief specifies.
NODE_TEXT_METHOD = "T04_raw_node_hybrid"

UNIT_OF = {
    "T04_summary_node_hybrid": "node",
    "T04_raw_node_hybrid": "node",
    "T04_raw_article_hybrid": "article",
}
SHORT = {
    "T04_summary_node_hybrid": "summary_node",
    "T04_raw_node_hybrid": "raw_node",
    "T04_raw_article_hybrid": "raw_article",
}


# ---------------------------------------------------------------------------
# Tokeniser (lazy, cached)
# ---------------------------------------------------------------------------

_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _TOK


def _tokens(text: str) -> list[str]:
    return _tokenizer().tokenize(text or "")


# ---------------------------------------------------------------------------
# Persisted-artifact resolution (rankings hash ⇒ matching faiss_meta)
# ---------------------------------------------------------------------------

def _resolve_variant_hashes(stem: str) -> dict[str, str]:
    """{variant_label: config_hash} using the SAME selection as load_t04_variants.

    Highest ``linker_version`` then most recent ``compare_*.jsonl`` mtime, ties to
    the earliest hash dir — so the faiss_meta we read for node text / linking is
    the one that produced the rankings we score.
    """
    base = paths.T04_BASE / stem
    results_root = base / "results"
    best: dict[str, tuple[int, float, str]] = {}
    for hd in sorted(p for p in results_root.iterdir() if p.is_dir()):
        js = sorted(hd.glob("compare_*.jsonl"))
        mf = base / "configs" / hd.name / "manifest.json"
        if not js or not mf.exists():
            continue
        ch = json.loads(mf.read_text(encoding="utf-8")).get("chunking", {})
        fusion = "sparse" if ch.get("sparse_only") else "hybrid"
        label = f"T04_{ch.get('variant')}_{ch.get('unit')}_{fusion}"
        linker = int(ch.get("linker_version", 0))
        mtime = js[-1].stat().st_mtime
        prev = best.get(label)
        if prev is not None and (linker, mtime) <= (prev[0], prev[1]):
            continue
        best[label] = (linker, mtime, hd.name)
    return {label: t[2] for label, t in best.items()}


def _load_faiss_meta(stem: str, config_hash: str) -> list[dict]:
    fp = paths.T04_BASE / stem / "configs" / config_hash / "faiss_meta.json"
    return json.loads(fp.read_text(encoding="utf-8"))["metadata"]


def _present_bsard(meta: list[dict]) -> set[int]:
    out: set[int] = set()
    for m in meta:
        for b in m.get("bsard_ids") or []:
            out.add(int(b))
    return out


def _bsard_db_text(stem: str) -> dict[int, str]:
    """{bsard_id: article_text} for this PDF, from the BSARD corpus DB."""
    con = sqlite3.connect(str(paths.BSARD_DB))
    try:
        fn = f"img_l_pdf_{stem}_F.pdf"
        rows = con.execute(
            "SELECT bsard_id, article_text FROM articles "
            "WHERE pdf_filename = ? AND bsard_id IS NOT NULL",
            (fn,),
        ).fetchall()
    finally:
        con.close()
    return {int(b): str(t or "") for b, t in rows}


# ---------------------------------------------------------------------------
# Per-stem assembly
# ---------------------------------------------------------------------------

@dataclass
class StemData:
    stem: str
    label: str
    gt: dict[str, list[int]]
    results: dict[str, dict[str, object]]      # variant -> {qid: RetrievalResult}
    node_table: dict[int, dict]                # node_id -> {"text", "bsard_ids"}
    present: dict[str, set[int]]               # variant -> bsard present in its index
    db_text: dict[int, str]                    # bsard_id -> DB article text


def load_stem(stem: str) -> StemData:
    label = paths.stem_label(stem)
    gt = loaders.load_ground_truth(stem)
    all_variants = loaders.load_t04_variants(stem)
    results = {
        v: {r.query_id: r for r in all_variants[v]}
        for v in VARIANTS if v in all_variants
    }
    hashes = _resolve_variant_hashes(stem)

    node_meta = _load_faiss_meta(stem, hashes[NODE_TEXT_METHOD])
    node_table = {
        int(m["node_id"]): {"text": m.get("text", ""),
                            "bsard_ids": [int(b) for b in (m.get("bsard_ids") or [])]}
        for m in node_meta
    }
    present = {NODE_TEXT_METHOD: _present_bsard(node_meta)}
    present["T04_summary_node_hybrid"] = present[NODE_TEXT_METHOD]   # same node set
    for v in ARTICLE_VARIANTS:
        if v in hashes:
            present[v] = _present_bsard(_load_faiss_meta(stem, hashes[v]))

    return StemData(stem, label, gt, results, node_table, present, _bsard_db_text(stem))


def node_ranking(result) -> list[int]:
    """Retrieved node_ids ordered by fused score desc (from ``metadata.node_id``)."""
    rows = [(int(it["metadata"]["node_id"]), float(it["score"]))
            for it in result.ranked_items]
    rows.sort(key=lambda r: -r[1])
    return [nid for nid, _ in rows]


# ---------------------------------------------------------------------------
# Group B — node→article token-weight table (both weights)
# ---------------------------------------------------------------------------

@dataclass
class Weights:
    db: dict[tuple[int, int], float]          # (node_id, bsard_id) -> w_db_overlap
    share: dict[tuple[int, int], float]       # (node_id, bsard_id) -> w_node_share
    table: pd.DataFrame                        # persisted rows
    art_node_token_sum: dict[int, float]       # bsard_id -> Σ node tokens (indexed)
    db_coverage_by_article: dict[int, float]   # bsard_id -> Σ w_db over its nodes


def build_weights(sd: StemData) -> Weights:
    """Build w_db_overlap (primary) + w_node_share (alongside) for every
    (node, bsard_id) pair in the node index of ``sd``.

    w_db_overlap(n,B) = |tok(node) ∩multiset tok(DB article B)| / |tok(DB B)|.
    w_node_share(n,B) = tok(node) / Σ_{nodes mapping to B} tok(node).
    """
    # token stats per node
    n_node_tok: dict[int, int] = {}
    node_counter: dict[int, Counter] = {}
    nodes_of: dict[int, list[int]] = defaultdict(list)
    for nid, rec in sd.node_table.items():
        toks = _tokens(rec["text"])
        n_node_tok[nid] = len(toks)
        node_counter[nid] = Counter(toks)
        for b in rec["bsard_ids"]:
            nodes_of[b].append(nid)

    art_node_token_sum = {
        b: float(sum(n_node_tok[n] for n in ns)) for b, ns in nodes_of.items()
    }
    # DB article token stats (only for articles actually present in the index)
    art_counter: dict[int, Counter] = {}
    art_len: dict[int, int] = {}
    for b in nodes_of:
        if b in sd.db_text and sd.db_text[b].strip():
            t = _tokens(sd.db_text[b])
            art_counter[b] = Counter(t)
            art_len[b] = len(t)

    db: dict[tuple[int, int], float] = {}
    share: dict[tuple[int, int], float] = {}
    rows: list[dict] = []
    db_cov: dict[int, float] = defaultdict(float)
    for b, ns in nodes_of.items():
        denom_share = art_node_token_sum[b]
        for nid in ns:
            w_share = n_node_tok[nid] / denom_share if denom_share else np.nan
            share[(nid, b)] = w_share
            if b in art_len and art_len[b]:
                overlap = sum((node_counter[nid] & art_counter[b]).values())
                w_db = overlap / art_len[b]
            else:
                w_db = np.nan
            db[(nid, b)] = w_db
            if not np.isnan(w_db):
                db_cov[b] += w_db
            rows.append({
                "stem": sd.stem, "node_id": nid, "bsard_id": b,
                "n_node_tokens": n_node_tok[nid],
                "w_db_overlap": w_db, "w_node_share": w_share,
            })
    table = pd.DataFrame(rows)
    return Weights(db, share, table, art_node_token_sum, dict(db_cov))


def _coverage(node_ids: list[int], wmap: dict[tuple[int, int], float],
              bsard_id: int) -> float:
    s = 0.0
    for n in node_ids:
        w = wmap.get((n, bsard_id))
        if w is not None and not np.isnan(w):
            s += w
    return min(1.0, s)


def coverage_long_table(sd: StemData, weights: Weights) -> pd.DataFrame:
    """One row per (variant, stem, qid, gold_bsard_id) for the NODE variants.

    coverage_db@k / coverage_share@k = capped Σ of the respective node→article
    weight over the top-k retrieved nodes (= fraction of that gold article's text
    the generator receives). Whole-unit ``raw_article`` is handled separately
    (coverage ≡ 1.0 on a hit).
    """
    rows: list[dict] = []
    for variant in NODE_VARIANTS:
        if variant not in sd.results:
            continue
        for qid, gold_list in sd.gt.items():
            gold = {int(b) for b in gold_list}
            if not gold or qid not in sd.results[variant]:
                continue
            nodes = node_ranking(sd.results[variant][qid])
            for b in gold:
                row: dict = {
                    "variant": variant, "stem": sd.stem, "stem_label": sd.label,
                    "qid": qid, "bsard_id": b, "n_gold": len(gold),
                    "is_multi": len(gold) > 1,
                }
                for k in K_NODE:
                    topk = nodes[:k]
                    cdb = _coverage(topk, weights.db, b)
                    csh = _coverage(topk, weights.share, b)
                    row[f"coverage_db@{k}"] = cdb
                    row[f"coverage_share@{k}"] = csh
                    row[f"touched@{k}"] = float(cdb > 0.0 or csh > 0.0)
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Group A / D1 per-question table (all three variants, shared primitives)
# ---------------------------------------------------------------------------

def per_question_table(sd: StemData) -> pd.DataFrame:
    rows: list[dict] = []
    for variant in VARIANTS:
        if variant not in sd.results:
            continue
        unit = UNIT_OF[variant]
        red = "nodes" if unit == "node" else "units"
        for qid, gold_list in sd.gt.items():
            gold = {int(b) for b in gold_list}
            if not gold or qid not in sd.results[variant]:
                continue
            res = sd.results[variant][qid]
            art_rank = common.article_ranking(res)
            unit_rank = common.unit_ranking(res)
            row: dict = {
                "variant": variant, "unit": unit,
                "stem": sd.stem, "stem_label": sd.label, "qid": qid,
                "n_gold": len(gold), "is_multi": len(gold) > 1,
            }
            row.update(common.article_metrics(gold, art_rank, K_ARTICLE))
            row.update(common.rank_of_gold(gold, art_rank))
            row.update(common.unit_context_metrics(
                gold, unit_rank, K_NODE, redundancy_label=red))
            row["units_to_cover_gold"] = common.units_to_cover_gold(gold, unit_rank)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Failure lens 1 — entire misses & missed-gold classification
# ---------------------------------------------------------------------------

def missed_gold_table(sd: StemData, extraction: dict[int, dict]) -> pd.DataFrame:
    """One row per (variant, stem, qid, missed gold bsard_id).

    A gold article is "missed" when it is NOT in the article ranking within the
    persisted pool (top-100). Each miss is classified:
      ABSENT       — no node maps to that bsard_id in the variant's index
                     (an extraction / linking gap);
      BEYOND_POOL  — present in the index but ranked beyond the top-100.
    ``is_entire_miss`` flags questions with NO gold retrieved in the top-100.
    Each row carries the missed article's own extraction status for the
    miss-class × extraction-status cross-tab (ANALYSIS_PLAN C2).
    """
    rows: list[dict] = []
    for variant in VARIANTS:
        if variant not in sd.results:
            continue
        present = sd.present.get(variant, set())
        for qid, gold_list in sd.gt.items():
            gold = {int(b) for b in gold_list}
            if not gold or qid not in sd.results[variant]:
                continue
            art_rank = common.article_ranking(sd.results[variant][qid])
            pool = set(art_rank[:POOL_DEPTH])
            retrieved_gold = gold & pool
            is_entire_miss = len(retrieved_gold) == 0
            amap = extraction.get(int(qid), {})
            for b in gold - pool:
                st, cos = amap.get(int(b), (None, None))
                rows.append({
                    "variant": variant, "unit": UNIT_OF[variant],
                    "stem": sd.stem, "stem_label": sd.label, "qid": qid,
                    "bsard_id": b, "n_gold": len(gold), "is_multi": len(gold) > 1,
                    "miss_class": "ABSENT" if b not in present else "BEYOND_POOL",
                    "article_status": st or "UNKNOWN",
                    "article_cosine": cos if cos is not None else np.nan,
                    "is_entire_miss": is_entire_miss,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _e_lookup(long: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    sub = long[long["method"].isin(VARIANTS)]
    return {(r["method"], r["stem"], r["metric"]): r["value"] for _, r in sub.iterrows()}


def per_pdf_summary(pq: pd.DataFrame, cov: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    e = _e_lookup(long)
    rows: list[dict] = []
    for (variant, stem), g in pq.groupby(["variant", "stem"]):
        row: dict = {
            "variant": variant, "variant_short": SHORT[variant], "unit": g["unit"].iloc[0],
            "stem": stem, "stem_label": g["stem_label"].iloc[0], "n_q": len(g),
            "n_single": int((~g["is_multi"]).sum()), "n_multi": int(g["is_multi"].sum()),
        }
        for k in K_ARTICLE:
            row[f"recall@{k}"] = g[f"recall@{k}"].mean()
        row["hit@10"] = g["hit@10"].mean()
        row["E/recall@10"] = e.get((variant, stem, "E/recall@10"), np.nan)
        row["E/recall@100"] = e.get((variant, stem, "E/recall@100"), np.nan)
        for k in K_NODE:
            row[f"precision@{k}"] = g[f"precision@{k}"].mean()
            row[f"distinct_articles@{k}"] = g[f"distinct_articles@{k}"].mean()
        row["median_first_hit_rank"] = g["first_hit_rank"].median()
        row["p90_first_hit_rank"] = g["first_hit_rank"].quantile(0.90)
        row["jaccard@10"] = g["jaccard@10"].mean()
        row["units_to_cover_gold_median"] = g["units_to_cover_gold"].median()
        # Group B coverage (node variants only); DB-overlap primary
        if not cov.empty and variant in NODE_VARIANTS:
            cg = cov[(cov["variant"] == variant) & (cov["stem"] == stem)]
            t10 = cg[cg["touched@10"] > 0]
            row["coverage_db@10_all"] = cg["coverage_db@10"].mean() if not cg.empty else np.nan
            row["coverage_db@10_touched"] = t10["coverage_db@10"].mean() if not t10.empty else np.nan
            row["coverage_share@10_touched"] = t10["coverage_share@10"].mean() if not t10.empty else np.nan
            row["pct_fragment_db@10"] = (100.0 * (t10["coverage_db@10"] < FRAGMENT_TAU).mean()
                                         if not t10.empty else np.nan)
            row["pct_near_complete_db@10"] = (100.0 * (t10["coverage_db@10"] >= NEAR_COMPLETE_TAU).mean()
                                              if not t10.empty else np.nan)
        else:
            # whole-unit article variant: coverage ≡ 1.0 on a hit
            row["coverage_db@10_all"] = np.nan
            row["coverage_db@10_touched"] = 1.0
            row["coverage_share@10_touched"] = 1.0
            row["pct_fragment_db@10"] = 0.0
            row["pct_near_complete_db@10"] = 100.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", "n_q"], ascending=[True, False]).reset_index(drop=True)


def aggregate_summary(pq: pd.DataFrame, cov: pd.DataFrame, per_pdf: pd.DataFrame,
                      long: pd.DataFrame) -> pd.DataFrame:
    e = _e_lookup(long)
    rows: list[dict] = []
    for variant in VARIANTS:
        gpq = pq[pq["variant"] == variant]
        if gpq.empty:
            continue
        gpdf = per_pdf[per_pdf["variant"] == variant]
        stems = gpq["stem"].unique()
        e_stem = {s: (e.get((variant, s, "E/recall@10"), np.nan),
                      e.get((variant, s, "E/recall@100"), np.nan),
                      e.get((variant, s, "E/n_evaluable"), np.nan)) for s in stems}

        def make(kind: str) -> dict:
            src = gpq if kind == "micro" else gpdf
            row: dict = {"variant": variant, "variant_short": SHORT[variant],
                         "unit": gpq["unit"].iloc[0], "aggregation": kind,
                         "n_q": len(gpq), "n_pdfs": gpq["stem"].nunique()}
            for k in K_ARTICLE:
                row[f"recall@{k}"] = src[f"recall@{k}"].mean()
            base = gpq["recall@10"].to_numpy() if kind == "micro" else gpdf["recall@10"].to_numpy()
            lo, hi = common.bootstrap_ci(base)
            row["recall@10_ci_lo"], row["recall@10_ci_hi"] = lo, hi
            row["hit@10"] = src["hit@10"].mean()
            if kind == "macro":
                row["E/recall@10"] = np.nanmean([v[0] for v in e_stem.values()])
                row["E/recall@100"] = np.nanmean([v[1] for v in e_stem.values()])
            else:
                num10 = num100 = den = 0.0
                for v10, v100, n in e_stem.values():
                    if not np.isnan(v10) and not np.isnan(n):
                        num10 += v10 * n
                        num100 += (v100 if not np.isnan(v100) else 0.0) * n
                        den += n
                row["E/recall@10"] = num10 / den if den else np.nan
                row["E/recall@100"] = num100 / den if den else np.nan
            for k in K_NODE:
                row[f"precision@{k}"] = src[f"precision@{k}"].mean()
                row[f"distinct_articles@{k}"] = src[f"distinct_articles@{k}"].mean()
            row["median_first_hit_rank"] = (gpq["first_hit_rank"].median() if kind == "micro"
                                            else gpdf["median_first_hit_rank"].mean())
            row["jaccard@10"] = src["jaccard@10"].mean()
            # coverage
            if variant in NODE_VARIANTS and not cov.empty:
                cg = cov[cov["variant"] == variant]
                t = cg[cg["touched@10"] > 0]
                if kind == "micro":
                    row["coverage_db@10_all"] = cg["coverage_db@10"].mean()
                    row["coverage_db@10_touched"] = t["coverage_db@10"].mean()
                    row["coverage_share@10_touched"] = t["coverage_share@10"].mean()
                    row["pct_fragment_db@10"] = 100.0 * (t["coverage_db@10"] < FRAGMENT_TAU).mean()
                    row["pct_near_complete_db@10"] = 100.0 * (t["coverage_db@10"] >= NEAR_COMPLETE_TAU).mean()
                else:
                    row["coverage_db@10_all"] = gpdf["coverage_db@10_all"].mean()
                    row["coverage_db@10_touched"] = gpdf["coverage_db@10_touched"].mean()
                    row["coverage_share@10_touched"] = gpdf["coverage_share@10_touched"].mean()
                    row["pct_fragment_db@10"] = gpdf["pct_fragment_db@10"].mean()
                    row["pct_near_complete_db@10"] = gpdf["pct_near_complete_db@10"].mean()
            else:
                row["coverage_db@10_all"] = np.nan
                row["coverage_db@10_touched"] = 1.0
                row["coverage_share@10_touched"] = 1.0
                row["pct_fragment_db@10"] = 0.0
                row["pct_near_complete_db@10"] = 100.0
            return row

        rows.append(make("micro"))
        rows.append(make("macro"))
    return pd.DataFrame(rows)


def strata_summary(pq: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Recall@10 + coverage by cardinality and extraction_class, per variant (micro)."""
    rows: list[dict] = []
    cov_q = None
    if not cov.empty:
        cov_q = cov.groupby(["variant", "stem", "qid"]).agg(
            cov_db10=("coverage_db@10", "mean")).reset_index()
        cov_q["qid"] = cov_q["qid"].astype(str)

    def block(g: pd.DataFrame, variant: str, stratum: str, col: str):
        for val, gg in g.groupby(col):
            row = {
                "variant": variant, "variant_short": SHORT[variant],
                "stratum": stratum, "level": val, "n_q": len(gg),
                "recall@5": gg["recall@5"].mean(), "recall@10": gg["recall@10"].mean(),
                "recall@20": gg["recall@20"].mean(), "hit@10": gg["hit@10"].mean(),
                "median_first_hit_rank": gg["first_hit_rank"].median(),
                "precision@10": gg["precision@10"].mean(),
            }
            if cov_q is not None and variant in NODE_VARIANTS:
                m = gg[["stem", "qid"]].copy()
                m["qid"] = m["qid"].astype(str)
                m = m.merge(cov_q[cov_q["variant"] == variant], on=["stem", "qid"], how="left")
                row["coverage_db@10"] = m["cov_db10"].mean()
            else:
                row["coverage_db@10"] = 1.0 if variant in ARTICLE_VARIANTS else np.nan
            rows.append(row)

    for variant in VARIANTS:
        g = pq[pq["variant"] == variant]
        if g.empty:
            continue
        block(g, variant, "cardinality", "cardinality")
        block(g, variant, "extraction_class", "extraction_class")
    return pd.DataFrame(rows)


def entire_miss_summary(missed: pd.DataFrame, pq: pd.DataFrame) -> pd.DataFrame:
    """Per-variant miss profile: missed-gold split (ABSENT/BEYOND_POOL),
    miss-class × extraction-status cross-tab counts, and entire-miss question rate."""
    rows: list[dict] = []
    for variant in VARIANTS:
        gpq = pq[pq["variant"] == variant]
        gm = missed[missed["variant"] == variant] if not missed.empty else missed
        n_q = len(gpq)
        n_gold = int(gpq["n_gold"].sum())
        n_missed = len(gm)
        n_absent = int((gm["miss_class"] == "ABSENT").sum()) if n_missed else 0
        n_beyond = int((gm["miss_class"] == "BEYOND_POOL").sum()) if n_missed else 0
        # entire-miss questions: those appearing with is_entire_miss True for any miss row
        em_qids = set(gm[gm["is_entire_miss"]]["qid"]) if n_missed else set()
        row = {
            "variant": variant, "variant_short": SHORT[variant], "unit": UNIT_OF[variant],
            "n_q": n_q, "n_gold": n_gold,
            "n_gold_missed_top100": n_missed,
            "pct_gold_missed": 100.0 * n_missed / n_gold if n_gold else np.nan,
            "n_absent": n_absent, "n_beyond_pool": n_beyond,
            "pct_missed_absent": 100.0 * n_absent / n_missed if n_missed else np.nan,
            "n_entire_miss_q": len(em_qids),
            "pct_entire_miss_q": 100.0 * len(em_qids) / n_q if n_q else np.nan,
        }
        # miss-class × extraction-status (article-level) counts
        for status in ("FOUND", "PARTIAL", "NOT_FOUND", "UNKNOWN"):
            sg = gm[gm["article_status"] == status] if n_missed else gm
            row[f"missed_{status}"] = len(sg)
            row[f"missed_{status}_absent"] = int((sg["miss_class"] == "ABSENT").sum()) if len(sg) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def assert_published_recall(
    pq: pd.DataFrame, long: pd.DataFrame, *,
    anchor_tol: float = 1.5e-3, gross_tol: float = 1.0e-2,
) -> dict:
    """Assert the recomputed article recall@k reproduces the published T1/R@k.

    The recomputation reads the CURRENT on-disk rankings; the consolidated frame
    was built earlier, and the *node* indexes have since been resynced, so a
    handful of node-variant cells drift by a few ×1e-3 at deeper k (a couple of
    articles crossing the rank-k boundary among many near-tied node scores). The
    article-unit variant (raw_article) reproduces every cell exactly. To catch a
    genuine wrong-artifact / wrong-aggregation error while tolerating this
    immaterial drift we use **two gates**:

      * anchor gate — R@5 and R@100 (the depth-stable cells; observed drift
        ≤2.7e-4) must match to ``anchor_tol`` (1.5e-3). A stale linker-v3 index
        would shift every k including these anchors by ≈1e-2 and fail here.
      * gross gate — every cell must match to ``gross_tol`` (1e-2), catching any
        aggregation-logic error.

    Returns ``{max_dev, max_anchor_dev, n_drift, n_cells}``.
    """
    devs: list[tuple[str, str, int, float]] = []
    for variant in VARIANTS:
        for stem, g in pq[pq["variant"] == variant].groupby("stem"):
            for k in K_ARTICLE:
                pub = long[(long["method"] == variant) & (long["stem"] == stem)
                           & (long["metric"] == f"T1/R@{k}")]["value"]
                if pub.empty:
                    continue
                dev = abs(g[f"recall@{k}"].mean() - float(pub.iloc[0]))
                devs.append((variant, stem, k, dev))
    max_dev = max((d for *_, d in devs), default=0.0)
    anchor = [(v, s, k, d) for v, s, k, d in devs if k in (5, 100)]
    max_anchor_dev = max((d for *_, d in anchor), default=0.0)
    bad_anchor = [(v, s, k, d) for v, s, k, d in anchor if d >= anchor_tol]
    bad_gross = [(v, s, k, d) for v, s, k, d in devs if d >= gross_tol]
    assert not bad_anchor, f"R@5/R@100 anchor mismatch (>{anchor_tol}): {bad_anchor[:5]}"
    assert not bad_gross, f"R@k gross mismatch (>{gross_tol}): {bad_gross[:5]}"
    return {"max_dev": max_dev, "max_anchor_dev": max_anchor_dev,
            "n_drift": sum(1 for *_, d in devs if d > 1e-6), "n_cells": len(devs)}
