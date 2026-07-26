"""Arm 2B (PageIndex) retrieved-article deep-dive — compute engine.

Profiles the *retrieved articles* of Arm 2B (T05 vectorless LLM tree-navigation
over the Law→Chapter→Article ToC), along the properties that matter for
downstream answer generation, per PDF and aggregated over the curated 5-PDF set
(725 questions). Pure read from the persisted one-path artifacts — navigation is
NEVER rerun (STYLEGUIDE §6 / EVALUATION_METHODOLOGY §4 / brief reproducibility
rule). Single canonical config: the live ``results/`` snapshot (post-fix,
chapter-then-law padding) — there is no variant grid.

Why no coverage (Group B)
-------------------------
Arm 2B retrieves WHOLE ARTICLES (a tree leaf *is* an article; one ``bsard_id``
per ranked item, ``item.text`` = the article text). Coverage of a retrieved gold
article is therefore ≡ 1.0 on every hit by construction — the Arm-1/Arm-2A
token-coverage lens does not apply and **no coverage distribution is computed**.
The Arm-2B analogue of "retrieval fraction of the correct articles" is the
NAVIGATED-vs-PADDED decomposition (Group N), which replaces it.

The three observable tiers of the result list (the Arm-2B lens)
---------------------------------------------------------------
The persisted ``ranked_items`` is the LLM's navigation output followed by
deterministic padding to ``pad_to_k=100`` (navigator.py). Reading the navigator,
three tiers are *observable* per query:

  * **navigated set** ``N_nav`` — the trace's ``summary.candidate_article_ids``:
    the deduped union of ``selected_article_ids`` over the article-selection steps
    (+ any ``follow_refs`` additions). This is the brief's "navigated set" — every
    article the LLM tree-walk actually reached and chose.
  * **exposed head** ``N_exp`` — the ranked items with ``score > 0``. By
    construction ``N_exp ⊆ N_nav``: a navigated article keeps its LLM
    ``evaluate`` score (or the default ``score_threshold=1`` when the model forgot
    to score it) and survives iff that score ≥ 1; an article the model explicitly
    scored 0 is *dropped from the head* and re-absorbed into the score-0 padded
    tail. ``N_exp`` is what sits ABOVE the padding — what a small-top-k generator
    actually receives as "the LLM's picks".
  * **padded tail** — ranked items with ``score == 0``: the deterministic
    chapter-then-law fill, which also re-absorbs evaluate-rejected candidates.

We label the recall senses explicitly (the Arm-1/2A lesson — never conflate a
"head" recall with a padded recall):
  * ``nav_recall``    — recall over ``N_nav``        (what navigation reached)
  * ``exposed_recall``— recall over ``N_exp``        (what survived to the head)
  * ``recall@k``      — recall over the full top-k   (padding-inflated at large k)

Navigated/padded boundary — reconstruction + validation
--------------------------------------------------------
``N_nav`` is read directly from the persisted trace ``summary`` (the navigator
recorded its own deduped candidate set — no fragile re-derivation needed). It is
cross-checked two ways and the agreement is reported:
  (a) ``N_exp == N_nav`` rate (they agree iff the evaluate step rejected nothing),
  (b) ``N_nav`` vs the RAW union of the per-step ``selected_article_ids`` (the
      trace records the model's pre-cap / pre-validity selections, so the raw
      union is a noisy OVER-count — documented, not used).
``N_exp`` is recomputed independently from the ranked-item scores.

Navigation-failure anatomy (ANALYSIS_PLAN C3, tree + trace)
-----------------------------------------------------------
Each gold article NOT in ``N_nav`` is classified via the tree + the selected
chapters:
  * ``ABSENT_FROM_TREE``           — no tree node carries that bsard_id
                                     (extraction / tree-build gap; C3-iii subset);
  * ``WRONG_CHAPTER``              — in the tree, but its chapter ∉ the LLM's
                                     ``selected_chapter_ids`` (C3-i: wrong subtree);
  * ``CHAPTER_OK_ARTICLE_MISSED``  — chapter selected, article not picked/kept
                                     (C3-ii: right chapter, wrong article).
Each row also records where the gold *landed* in the result (exposed head /
padded tail / outside the top-100), and its extraction status.

Determinism: seed = 42 for the only stochastic step (bootstrap CIs). Idempotent.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import deep_dive_common as common
from . import loaders, paths

# --- config ----------------------------------------------------------------
K_ARTICLE = [5, 10, 20, 100]
PRIMARY_K = 10
POOL_DEPTH = 100                # T05 pads to pad_to_k=100
SEED = common.SEED
N_BOOT = common.N_BOOT
METHOD = "T05_pageindex"        # the consolidated long-frame method name
SNAPSHOT = "results"            # live post-fix + chapter-then-law padding

# navigation-failure classes (ANALYSIS_PLAN C3)
NAV_CLASSES = ["WRONG_CHAPTER", "CHAPTER_OK_ARTICLE_MISSED", "ABSENT_FROM_TREE"]
LANDING_TIERS = ["exposed_head", "padded_tail", "outside_pool"]


# ---------------------------------------------------------------------------
# Minimal RetrievalResult shim so the SHARED deep_dive_common primitives apply
# (each ranked item carries metadata.bsard_ids = [bsard_id]; one article/item).
# ---------------------------------------------------------------------------

@dataclass
class _Res:
    ranked_items: list[dict]


def _bid_from_node_id(node_id: str) -> int | None:
    """``ART_doc_9_bsard_960`` -> 960 (used only as a fallback)."""
    try:
        return int(str(node_id).rsplit("_", 1)[-1])
    except (ValueError, AttributeError):
        return None


def _make_result(raw: dict) -> _Res:
    """Wrap a raw T05 query JSON as a shared-primitive-compatible result.

    Normalises ``metadata.bsard_id`` (singular, the T05 contract) to the
    ``bsard_ids`` list the shared ``article_ranking`` expects. Ranked items are
    already deduped to one bsard_id each (retriever.py), and already score-sorted
    with the score-0 padding last, so the article ranking is order-faithful to
    the persisted list — the same list T07 scored for the published T1/R@k.
    """
    items: list[dict] = []
    for it in raw.get("ranked_items", []):
        md = it.get("metadata") or {}
        b = md.get("bsard_id")
        if b is None:
            continue
        items.append({
            "id": it.get("id"),
            "score": float(it.get("score", 0.0)),
            "metadata": {"bsard_ids": [int(b)]},
        })
    return _Res(items)


def _exposed_set(raw: dict) -> set[int]:
    """Exposed head N_exp = bsard ids of ranked items with score > 0."""
    out: set[int] = set()
    for it in raw.get("ranked_items", []):
        if float(it.get("score", 0.0)) > 0.0:
            b = (it.get("metadata") or {}).get("bsard_id")
            if b is not None:
                out.add(int(b))
    return out


# ---------------------------------------------------------------------------
# Tree: bsard_id -> chapter (direct child of the law), present-set, chapter map
# ---------------------------------------------------------------------------

@dataclass
class TreeMaps:
    bsard_to_chapter: dict[int, str]      # bsard_id -> chapter node_id
    node_to_bsard: dict[str, int]         # leaf node_id -> bsard_id
    present: set[int]                     # all bsard_ids in the tree
    chapter_articles: dict[str, list[int]]  # chapter node_id -> [bsard_id]
    law_ids: list[str]


def _iter_leaves(node: dict):
    """Yield every leaf dict (bsard_id set, no sub_nodes) under ``node``."""
    kids = node.get("sub_nodes") or []
    if not kids:
        if node.get("bsard_id") is not None:
            yield node
        return
    for c in kids:
        yield from _iter_leaves(c)


def load_tree(stem: str) -> TreeMaps:
    """Build the bsard->chapter map from the persisted ``tree.json``.

    A "chapter" is a *direct child of the law node* — exactly the granularity the
    navigator's chapter-selection step picks from (navigator.select_chapters
    iterates ``law_node.sub_nodes``). For a 2-level law (children are leaves), the
    navigator synthesises the LAW node id as the chapter, so a direct-leaf
    article's chapter = the law node id (matches ``selected_chapter_ids``).
    """
    fp = paths.T05_BASE / stem / "tree.json"
    root = json.loads(fp.read_text(encoding="utf-8"))["tree"]

    # Resolve the law node(s): the curated PDFs are single-doc (root is LAW_*),
    # but tolerate a ROOT wrapper holding law children.
    if str(root.get("node_id", "")).startswith("LAW_"):
        laws = [root]
    else:
        laws = [n for n in (root.get("sub_nodes") or [])
                if str(n.get("node_id", "")).startswith("LAW_")] or [root]

    bsard_to_chapter: dict[int, str] = {}
    node_to_bsard: dict[str, int] = {}
    chapter_articles: dict[str, list[int]] = {}
    present: set[int] = set()

    for law in laws:
        law_id = law["node_id"]
        for child in (law.get("sub_nodes") or []):
            # 2-level law: a direct child is itself an article leaf -> chapter = law.
            if not (child.get("sub_nodes") or []) and child.get("bsard_id") is not None:
                chapter_id = law_id
            else:
                chapter_id = child["node_id"]
            for leaf in _iter_leaves(child):
                b = int(leaf["bsard_id"])
                bsard_to_chapter[b] = chapter_id
                node_to_bsard[leaf["node_id"]] = b
                present.add(b)
                chapter_articles.setdefault(chapter_id, []).append(b)
            # also handle the case where `child` itself is the leaf
            if not (child.get("sub_nodes") or []) and child.get("bsard_id") is not None:
                b = int(child["bsard_id"])
                bsard_to_chapter[b] = chapter_id
                node_to_bsard[child["node_id"]] = b
                present.add(b)
                chapter_articles.setdefault(chapter_id, []).append(b)

    return TreeMaps(bsard_to_chapter, node_to_bsard, present,
                    chapter_articles, [law["node_id"] for law in laws])


# ---------------------------------------------------------------------------
# Per-stem assembly
# ---------------------------------------------------------------------------

@dataclass
class StemData:
    stem: str
    label: str
    gt: dict[str, list[int]]
    raw: dict[str, dict]          # qid -> raw T05 query JSON (ranked_items+trace+cost)
    tree: TreeMaps


def load_stem(stem: str) -> StemData:
    label = paths.stem_label(stem)
    gt = loaders.load_ground_truth(stem)
    d_dir = paths.T05_BASE / stem / SNAPSHOT
    raw: dict[str, dict] = {}
    for fp in sorted(d_dir.glob("q*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        raw[str(d["query_id"])] = d
    return StemData(stem, label, gt, raw, load_tree(stem))


# ---------------------------------------------------------------------------
# Navigated / exposed reconstruction (per query)
# ---------------------------------------------------------------------------

def navigated_set(raw: dict, tree: TreeMaps) -> set[int]:
    """N_nav = bsard ids of the trace's ``summary.candidate_article_ids``.

    Maps each candidate node_id to a bsard_id via the tree (fallback: parse the
    ``_bsard_<n>`` suffix). This is the navigator's own deduped selection set.
    """
    out: set[int] = set()
    cand = (raw.get("trace") or {}).get("summary", {}).get("candidate_article_ids", [])
    for nid in cand:
        b = tree.node_to_bsard.get(nid)
        if b is None:
            b = _bid_from_node_id(nid)
        if b is not None:
            out.add(int(b))
    return out


def raw_selected_union(raw: dict, tree: TreeMaps) -> set[int]:
    """The RAW union of per-step ``selected_article_ids`` (a noisy over-count).

    The trace records the model's pre-cap / pre-validity selections, so this set
    is ≥ ``candidate_article_ids``. Reported only to quantify trace-vs-result
    agreement (the brief's "how cleanly the two methods agree"); never used for
    metrics.
    """
    out: set[int] = set()
    for s in (raw.get("trace") or {}).get("steps", []):
        if str(s.get("step", "")).startswith("article_selection"):
            for nid in (s.get("parsed") or {}).get("selected_article_ids", []) or []:
                b = tree.node_to_bsard.get(nid)
                if b is None:
                    b = _bid_from_node_id(nid)
                if b is not None:
                    out.add(int(b))
    return out


def selected_chapters(raw: dict) -> set[str]:
    return set((raw.get("trace") or {}).get("summary", {}).get("selected_chapter_ids", []))


# ---------------------------------------------------------------------------
# Group A (article level) + Group N (navigated vs exposed vs padded)
# ---------------------------------------------------------------------------

def per_question_table(sd: StemData) -> pd.DataFrame:
    rows: list[dict] = []
    for qid, gold_list in sd.gt.items():
        gold = {int(b) for b in gold_list}
        if not gold or qid not in sd.raw:
            continue
        raw = sd.raw[qid]
        res = _make_result(raw)
        art_rank = common.article_ranking(res)
        unit_rank = common.unit_ranking(res)
        exposed = _exposed_set(raw)
        navigated = navigated_set(raw, sd.tree)
        raw_union = raw_selected_union(raw, sd.tree)
        pool = set(art_rank[:POOL_DEPTH])
        n_exp = len(exposed)

        # Random-padding counterfactual: keep the exposed head, but fill the SAME
        # number of slots (out to POOL_DEPTH) with articles drawn uniformly at
        # random from the document instead of the LLM's selected chapters-then-
        # laws. Expected recall is exact (hypergeometric mean) -- no simulation.
        # Isolates how much of padded_recall@100 is genuine chapter localisation
        # vs. a pure coverage artifact (returning a big fraction of a small doc).
        n_doc = len(sd.tree.present)
        hits_head = len(gold & exposed)
        g_rem = len((gold & sd.tree.present) - exposed)   # gold catchable by fill
        non_head = n_doc - n_exp
        m_fill = max(0, min(POOL_DEPTH, n_doc) - n_exp)
        e_caught = (g_rem * m_fill / non_head) if non_head > 0 else 0.0

        cost = raw.get("cost", {}) or {}
        row: dict = {
            "stem": sd.stem, "stem_label": sd.label, "qid": qid,
            "n_gold": len(gold), "is_multi": len(gold) > 1,
            # --- Group A (shared primitives; k = articles) ---
        }
        row.update(common.article_metrics(gold, art_rank, K_ARTICLE))
        row.update(common.rank_of_gold(gold, art_rank))
        # Arm 2B unit IS an article -> precision@k is standard precision@k,
        # distinct_articles@k = k (no within-unit redundancy).
        row.update(common.unit_context_metrics(
            gold, unit_rank, K_ARTICLE, redundancy_label="articles"))

        # --- Group N: navigated vs exposed vs padded ---
        first = row["first_hit_rank"]
        row.update({
            "n_navigated": len(navigated),
            "n_exposed": n_exp,
            "n_raw_selected": len(raw_union),
            "exp_eq_nav": exposed == navigated,            # boundary agreement (a)
            "nav_recall": len(gold & navigated) / len(gold),
            "exposed_recall": len(gold & exposed) / len(gold),
            # padded recall == recall@100 (full pool); name it for clarity
            "padded_recall@100": len(gold & pool) / len(gold),
            # expected recall@100 if the same #fillers were drawn at random
            "random_recall@100": (hits_head + e_caught) / len(gold),
            "n_doc_articles": n_doc,
            "nav_precision": (len(gold & navigated) / len(navigated)
                              if navigated else np.nan),
            "exposed_precision": (len(gold & exposed) / n_exp if n_exp else np.nan),
            # exposed items occupy exactly the top n_exp ranks (score>0 sort first)
            "first_gold_in_exposed": bool(
                not np.isnan(first) and first <= n_exp) if n_exp else False,
            "first_gold_padded_only": bool(
                not np.isnan(first) and first > n_exp),
            "hit@100": float(len(gold & pool) > 0),
            # --- cost (no latency) ---
            "llm_calls": cost.get("llm_calls", np.nan),
            "tokens_in": cost.get("tokens_in", np.nan),
            "tokens_out": cost.get("tokens_out", np.nan),
            "exit_reason": cost.get("exit_reason", "unknown"),
            "iterations": cost.get("iterations", np.nan),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Navigation-failure anatomy (one row per gold NOT in the navigated set)
# ---------------------------------------------------------------------------

def _classify_nav_failure(b: int, tree: TreeMaps, sel_chapters: set[str],
                          sel_laws: set[str]) -> str:
    if b not in tree.present:
        return "ABSENT_FROM_TREE"
    ch = tree.bsard_to_chapter.get(b)
    if ch in sel_chapters or ch in sel_laws:
        return "CHAPTER_OK_ARTICLE_MISSED"
    return "WRONG_CHAPTER"


def _landing_tier(b: int, exposed: set[int], pool: set[int]) -> str:
    if b in exposed:
        return "exposed_head"
    if b in pool:
        return "padded_tail"
    return "outside_pool"


def nav_anatomy_table(sd: StemData, extraction: dict[int, dict]) -> pd.DataFrame:
    """One row per (qid, gold bsard) NOT reached by navigation.

    ``nav_class`` is the C3 anatomy; ``landing_tier`` is where the gold ended up
    in the result; ``article_status`` carries its extraction status.
    """
    rows: list[dict] = []
    for qid, gold_list in sd.gt.items():
        gold = {int(b) for b in gold_list}
        if not gold or qid not in sd.raw:
            continue
        raw = sd.raw[qid]
        res = _make_result(raw)
        art_rank = common.article_ranking(res)
        pool = set(art_rank[:POOL_DEPTH])
        exposed = _exposed_set(raw)
        navigated = navigated_set(raw, sd.tree)
        sel_ch = selected_chapters(raw)
        sel_laws = set((raw.get("trace") or {}).get("summary", {}).get("selected_law_ids", []))
        amap = extraction.get(int(qid), {})
        for b in gold - navigated:           # gold the navigation did not reach
            st, cos = amap.get(int(b), (None, None))
            rows.append({
                "stem": sd.stem, "stem_label": sd.label, "qid": qid,
                "bsard_id": b, "n_gold": len(gold), "is_multi": len(gold) > 1,
                "nav_class": _classify_nav_failure(b, sd.tree, sel_ch, sel_laws),
                "landing_tier": _landing_tier(b, exposed, pool),
                "chapter_id": sd.tree.bsard_to_chapter.get(b, "<absent>"),
                "article_status": st or "UNKNOWN",
                "article_cosine": cos if cos is not None else np.nan,
            })
    return pd.DataFrame(rows)


def gold_landing_table(sd: StemData) -> pd.DataFrame:
    """One row per (qid, gold bsard): landing tier + whether navigated.

    Drives failure lens 2 (entire misses) and the navigated-vs-padded headline:
    every gold article gets a tier (exposed_head / padded_tail / outside_pool)
    and a ``navigated`` flag.
    """
    rows: list[dict] = []
    for qid, gold_list in sd.gt.items():
        gold = {int(b) for b in gold_list}
        if not gold or qid not in sd.raw:
            continue
        raw = sd.raw[qid]
        res = _make_result(raw)
        art_rank = common.article_ranking(res)
        pool = set(art_rank[:POOL_DEPTH])
        exposed = _exposed_set(raw)
        navigated = navigated_set(raw, sd.tree)
        for b in gold:
            rows.append({
                "stem": sd.stem, "stem_label": sd.label, "qid": qid,
                "bsard_id": b, "n_gold": len(gold), "is_multi": len(gold) > 1,
                "landing_tier": _landing_tier(b, exposed, pool),
                "navigated": b in navigated,
                "exposed": b in exposed,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _e_lookup(long: pd.DataFrame) -> dict[tuple[str, str], float]:
    sub = long[long["method"] == METHOD]
    return {(r["stem"], r["metric"]): r["value"] for _, r in sub.iterrows()}


def per_pdf_summary(pq: pd.DataFrame, landing: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    e = _e_lookup(long)
    rows: list[dict] = []
    for stem, g in pq.groupby("stem"):
        row: dict = {
            "stem": stem, "stem_label": g["stem_label"].iloc[0], "n_q": len(g),
            "n_single": int((~g["is_multi"]).sum()), "n_multi": int(g["is_multi"].sum()),
        }
        for k in K_ARTICLE:
            row[f"recall@{k}"] = g[f"recall@{k}"].mean()
        row["hit@10"] = g["hit@10"].mean()
        row["E/recall@10"] = e.get((stem, "E/recall@10"), np.nan)
        row["E/recall@100"] = e.get((stem, "E/recall@100"), np.nan)
        # Group N
        row["nav_recall"] = g["nav_recall"].mean()
        row["exposed_recall"] = g["exposed_recall"].mean()
        row["padded_recall@100"] = g["padded_recall@100"].mean()
        row["random_recall@100"] = g["random_recall@100"].mean()
        row["nav_precision"] = g["nav_precision"].mean()
        row["exposed_precision"] = g["exposed_precision"].mean()
        row["precision@10"] = g["precision@10"].mean()
        row["n_navigated_median"] = g["n_navigated"].median()
        row["n_exposed_median"] = g["n_exposed"].median()
        row["exp_eq_nav_rate"] = g["exp_eq_nav"].mean()
        # of HITS (gold in pool), fraction whose first gold is in the exposed head
        hits = g[g["hit@100"] > 0]
        row["pct_hit_first_in_exposed"] = (100.0 * hits["first_gold_in_exposed"].mean()
                                           if not hits.empty else np.nan)
        row["median_first_hit_rank"] = g["first_hit_rank"].median()
        row["p90_first_hit_rank"] = g["first_hit_rank"].quantile(0.90)
        row["jaccard@10"] = g["jaccard@10"].mean()
        # cost (no latency)
        row["llm_calls_mean"] = g["llm_calls"].mean()
        row["tokens_per_q_mean"] = (g["tokens_in"].fillna(0) + g["tokens_out"].fillna(0)).mean()
        # landing of all gold
        lg = landing[landing["stem"] == stem]
        n_gold = len(lg)
        for tier in LANDING_TIERS:
            row[f"pct_gold_{tier}"] = 100.0 * (lg["landing_tier"] == tier).mean() if n_gold else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_q", ascending=False).reset_index(drop=True)


def aggregate_summary(pq: pd.DataFrame, per_pdf: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    e = _e_lookup(long)
    stems = pq["stem"].unique()
    e_stem = {s: (e.get((s, "E/recall@10"), np.nan),
                  e.get((s, "E/recall@100"), np.nan),
                  e.get((s, "E/n_evaluable"), np.nan)) for s in stems}
    rows: list[dict] = []

    def make(kind: str) -> dict:
        src = pq if kind == "micro" else per_pdf
        row: dict = {"aggregation": kind, "n_q": len(pq), "n_pdfs": pq["stem"].nunique()}
        for k in K_ARTICLE:
            row[f"recall@{k}"] = src[f"recall@{k}"].mean()
        base = (pq["recall@10"].to_numpy() if kind == "micro"
                else per_pdf["recall@10"].to_numpy())
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
        # Group N
        for col in ("nav_recall", "exposed_recall", "padded_recall@100",
                    "random_recall@100",
                    "nav_precision", "exposed_precision", "precision@10",
                    "jaccard@10"):
            row[col] = src[col].mean()
        if kind == "micro":
            row["n_navigated_median"] = pq["n_navigated"].median()
            row["n_exposed_median"] = pq["n_exposed"].median()
            row["exp_eq_nav_rate"] = pq["exp_eq_nav"].mean()
            hits = pq[pq["hit@100"] > 0]
            row["pct_hit_first_in_exposed"] = (100.0 * hits["first_gold_in_exposed"].mean()
                                               if not hits.empty else np.nan)
            row["median_first_hit_rank"] = pq["first_hit_rank"].median()
            row["llm_calls_mean"] = pq["llm_calls"].mean()
            row["tokens_per_q_mean"] = (pq["tokens_in"].fillna(0)
                                        + pq["tokens_out"].fillna(0)).mean()
        else:
            row["n_navigated_median"] = per_pdf["n_navigated_median"].mean()
            row["n_exposed_median"] = per_pdf["n_exposed_median"].mean()
            row["exp_eq_nav_rate"] = per_pdf["exp_eq_nav_rate"].mean()
            row["pct_hit_first_in_exposed"] = per_pdf["pct_hit_first_in_exposed"].mean()
            row["median_first_hit_rank"] = per_pdf["median_first_hit_rank"].mean()
            row["llm_calls_mean"] = per_pdf["llm_calls_mean"].mean()
            row["tokens_per_q_mean"] = per_pdf["tokens_per_q_mean"].mean()
        return row

    rows.append(make("micro"))
    rows.append(make("macro"))
    return pd.DataFrame(rows)


def strata_summary(pq: pd.DataFrame) -> pd.DataFrame:
    """Recall@k + Group N by cardinality and extraction_class (micro)."""
    rows: list[dict] = []

    def block(g: pd.DataFrame, stratum: str, col: str):
        for val, gg in g.groupby(col):
            rows.append({
                "stratum": stratum, "level": val, "n_q": len(gg),
                "recall@5": gg["recall@5"].mean(), "recall@10": gg["recall@10"].mean(),
                "recall@20": gg["recall@20"].mean(), "recall@100": gg["recall@100"].mean(),
                "hit@10": gg["hit@10"].mean(),
                "nav_recall": gg["nav_recall"].mean(),
                "exposed_recall": gg["exposed_recall"].mean(),
                "median_first_hit_rank": gg["first_hit_rank"].median(),
                "precision@10": gg["precision@10"].mean(),
                "n_exposed_median": gg["n_exposed"].median(),
            })

    block(pq, "cardinality", "cardinality")
    block(pq, "extraction_class", "extraction_class")
    return pd.DataFrame(rows)


def nav_anatomy_summary(anat: pd.DataFrame, landing: pd.DataFrame,
                        pq: pd.DataFrame) -> pd.DataFrame:
    """Per-PDF + aggregate navigation-failure anatomy.

    Denominator = all gold articles; counts split into NAVIGATED (reached) and
    the three failure classes. Also the landing of the not-navigated gold.
    """
    rows: list[dict] = []
    stems = list(pq["stem"].unique())

    def block(label_stem: str | None):
        if label_stem is None:
            a, lg = anat, landing
            stem_label, n_q = "ALL (aggregate)", int(len(pq))
        else:
            a = anat[anat["stem"] == label_stem]
            lg = landing[landing["stem"] == label_stem]
            stem_label = paths.stem_label(label_stem)
            n_q = int((pq["stem"] == label_stem).sum())
        n_gold = len(lg)
        n_navigated = int(lg["navigated"].sum())
        row = {
            "stem": label_stem or "ALL", "stem_label": stem_label, "n_q": n_q,
            "n_gold": n_gold,
            "n_navigated": n_navigated,
            "pct_navigated": 100.0 * n_navigated / n_gold if n_gold else np.nan,
            "n_not_navigated": len(a),
        }
        for cls in NAV_CLASSES:
            n = int((a["nav_class"] == cls).sum()) if not a.empty else 0
            row[f"n_{cls}"] = n
            row[f"pct_{cls}"] = 100.0 * n / n_gold if n_gold else np.nan
        # landing of all gold
        for tier in LANDING_TIERS:
            n = int((lg["landing_tier"] == tier).sum()) if n_gold else 0
            row[f"n_{tier}"] = n
            row[f"pct_{tier}"] = 100.0 * n / n_gold if n_gold else np.nan
        rows.append(row)

    for stem in sorted(stems, key=lambda s: -(pq["stem"] == s).sum()):
        block(stem)
    block(None)
    return pd.DataFrame(rows)


def entire_miss_summary(landing: pd.DataFrame, pq: pd.DataFrame) -> pd.DataFrame:
    """Failure lens 2 — question-level entire-miss profile.

    A question is an ENTIRE MISS when no gold article lands in the top-100.
    Cross with cardinality and extraction_class; report where its gold sits.
    """
    # question-level: did any gold reach the pool / exposed head?
    ql = landing.groupby(["stem", "qid"]).agg(
        any_pool=("landing_tier", lambda s: (s != "outside_pool").any()),
        any_exposed=("exposed", "any"),
        any_navigated=("navigated", "any"),
    ).reset_index()
    ql["qid"] = ql["qid"].astype(str)
    pqx = pq[["stem", "qid", "is_multi", "cardinality", "extraction_class"]].copy()
    pqx["qid"] = pqx["qid"].astype(str)
    m = pqx.merge(ql, on=["stem", "qid"], how="left")
    m["entire_miss"] = ~m["any_pool"].fillna(False)

    rows: list[dict] = []

    def block(stratum: str, col: str | None):
        groups = [("all", m)] if col is None else list(m.groupby(col))
        for level, gg in groups:
            n = len(gg)
            rows.append({
                "stratum": stratum, "level": level, "n_q": n,
                "pct_entire_miss": 100.0 * gg["entire_miss"].mean() if n else np.nan,
                "pct_any_exposed": 100.0 * gg["any_exposed"].fillna(False).mean() if n else np.nan,
                "pct_any_navigated": 100.0 * gg["any_navigated"].fillna(False).mean() if n else np.nan,
            })

    block("overall", None)
    block("cardinality", "cardinality")
    block("extraction_class", "extraction_class")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reproducibility gate — recomputed article R@k vs published T05 T1/R@k
# ---------------------------------------------------------------------------

def assert_published_recall(pq: pd.DataFrame, long: pd.DataFrame, *,
                            tol: float = 1.0e-9) -> dict:
    """Assert recomputed article recall@k reproduces the published T1/R@k for T05.

    Arm 2B's article ranking is deterministic (one bsard per item, score-sorted
    with score-0 padding last) and the ``results/`` snapshot has not been resynced
    since the consolidated frame was built, so we expect an EXACT match (unlike
    the node-variant drift the Arm-2A dive tolerated). Returns
    ``{max_dev, n_cells}`` and raises on any cell exceeding ``tol``.
    """
    devs: list[tuple[str, int, float]] = []
    for stem, g in pq.groupby("stem"):
        for k in K_ARTICLE:
            pub = long[(long["method"] == METHOD) & (long["stem"] == stem)
                       & (long["metric"] == f"T1/R@{k}")]["value"]
            if pub.empty:
                continue
            dev = abs(g[f"recall@{k}"].mean() - float(pub.iloc[0]))
            devs.append((stem, k, dev))
    max_dev = max((d for *_, d in devs), default=0.0)
    bad = [(s, k, d) for s, k, d in devs if d > tol]
    assert not bad, f"Arm-2B R@k mismatch vs published T1/R@k (>{tol}): {bad[:8]}"
    return {"max_dev": max_dev, "n_cells": len(devs)}
