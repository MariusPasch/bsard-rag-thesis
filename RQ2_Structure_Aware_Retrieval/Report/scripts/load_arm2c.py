"""Tidy loaders for the Arm-2C (agentic deep-tree navigation) experiment.

Arm 2C is the follow-up to Arm 2B (PageIndex): the flat ToC tree is rebuilt to
its full depth from AzureDI header-stacks and navigated by a ReAct agent
(vectorless LLaMA 3.1 8B), with native-metadata enrichment and a final LLM
re-rank. It was run *outside* the T06 consolidation, so — unlike Arms 1/2A/2B —
its numbers do NOT live in ``cross_arm_long.csv``. This module is the single
place that knows the on-disk schema of the arm2c run artefacts, mirroring
``load_results.py`` for the T06 frames.

Data source (the experiment's committed runs):
  RQ2_T04_ARM2_METADATA/experiments/arm2c/runs/
    arm2c_<stem>_enriched_rerank/per_query.csv   -- the headline config, 5 PDFs
    arm2c_1804_..._{bare,enriched,enriched_rerank}/per_query.csv  -- 1804 ablation

per_query.csv columns: qid, n_gold, recall@10, [recall@10_pre], recall@100,
hit@10, mrr@10, n_selected, nodes_visited, llm_calls, exit, gold_HIT,
gold_SEEN_NOT_SELECTED, gold_NOT_REACHED, gold_ORPHAN_UNREACHED.

Cross-arm comparison rule (STYLEGUIDE §6): Arms 1/2A/2B are read from the T06
single-re-eval long frame (``load_results``), never from the arm2c bundle's own
baselines.json — so the comparison stays apples-to-apples. The arm2c run's
per-PDF question set is the identical clipped-GT set, so the join is on ``stem``.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import load_results as lr  # noqa: E402
import thesis_style as ts  # noqa: E402

# --- arm2c run location ---------------------------------------------------
ARM2C_DIR = (ts.REPO_ROOT / "RQ2_T04_ARM2_METADATA" / "experiments" / "arm2c")
RUNS_DIR = ARM2C_DIR / "runs"

# Documents, in the canonical large-first display order, with the T06 stem_label
# so a merged frame reads the same as the cross-arm artefacts.
DOCS: list[tuple[str, str, str]] = [
    # (stem, T06 stem_label, short label for compact axes)
    ("1804_03_21_1804032150", "Code Civil",                "Code Civil"),
    ("1967_10_10_1967101055", "Code Judiciaire (larger)",  "Code Jud. (larger)"),
    ("2003_07_17_2013A31614", "Code du Logement",         "Code Logement"),
    ("1967_10_10_1967101056", "Code Judiciaire (smaller)", "Code Jud. (smaller)"),
    ("1867_06_08_1867060850", "Code Pénal",           "Code Pénal"),
]
STEM_LABEL = {s: lab for s, lab, _ in DOCS}
STEM_SHORT = {s: short for s, _, short in DOCS}

# The 1804 component ablation: deep tree alone -> + metadata -> + re-rank.
ABLATION_STEM = "1804_03_21_1804032150"
ABLATION_STEPS = [
    ("bare",            "Deep tree (labels only)"),
    ("enriched",        "+ metadata (enriched)"),
    ("enriched_rerank", "+ re-rank"),
]

MISS_CLASSES = ["gold_HIT", "gold_SEEN_NOT_SELECTED",
                "gold_NOT_REACHED", "gold_ORPHAN_UNREACHED"]


# ---------------------------------------------------------------------------
# Raw per-query reader
# ---------------------------------------------------------------------------

def _run_dir(stem: str, register: str = "enriched_rerank") -> Path:
    return RUNS_DIR / f"arm2c_{stem}_{register}"


def load_per_query(stem: str, register: str = "enriched_rerank") -> pd.DataFrame:
    """Per-query frame for one arm2c run (numeric columns coerced)."""
    p = _run_dir(stem, register) / "per_query.csv"
    if not p.exists():
        raise FileNotFoundError(f"arm2c run not found: {p}")
    df = pd.read_csv(p)
    for c in df.columns:
        if c != "exit":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Per-run aggregate (one row)
# ---------------------------------------------------------------------------

def _aggregate(df: pd.DataFrame) -> dict:
    n = len(df)
    gold_total = float(df["n_gold"].sum())
    miss = {c: float(df[c].sum()) for c in MISS_CLASSES}
    row = {
        "n": n,
        "R@10": df["recall@10"].mean(),
        "R@100": df["recall@100"].mean(),
        "Hit@10": df["hit@10"].mean(),
        "MRR@10": df["mrr@10"].mean(),
        "nodes_per_q": df["nodes_visited"].mean(),
        "llm_calls_per_q": df["llm_calls"].mean(),
        "selected_per_q": df["n_selected"].mean(),
        "empty_sel_q": int((df["n_selected"] == 0).sum()),
        "gold_total": gold_total,
        **{f"{c}_n": miss[c] for c in MISS_CLASSES},
        **{f"{c}_frac": (miss[c] / gold_total if gold_total else float("nan"))
           for c in MISS_CLASSES},
    }
    if "recall@10_pre" in df.columns:
        row["R@10_pre"] = df["recall@10_pre"].mean()
    return row


def arm2c_per_pdf(register: str = "enriched_rerank") -> pd.DataFrame:
    """One row per document for the given arm2c register, plus a weighted row."""
    rows = []
    for stem, label, short in DOCS:
        agg = _aggregate(load_per_query(stem, register))
        rows.append({"stem": stem, "stem_label": label, "short": short, **agg})
    df = pd.DataFrame(rows)

    # corpus-weighted (micro) row -- means weighted by question count, sums added.
    w = df["n"]
    wt: dict = {"stem": "corpus", "stem_label": "Corpus-weighted", "short": "Corpus"}
    for c in df.columns:
        if c in ("stem", "stem_label", "short"):
            continue
        if c == "n" or c.endswith("_n") or c == "gold_total" or c == "empty_sel_q":
            wt[c] = df[c].sum()
        elif c.endswith("_frac"):
            base = c[:-5] + "_n"
            wt[c] = df[base].sum() / df["gold_total"].sum()
        else:
            wt[c] = (df[c] * w).sum() / w.sum()
    return pd.concat([df, pd.DataFrame([wt])], ignore_index=True)


def arm2c_ablation() -> pd.DataFrame:
    """One row per ablation step on the Code Civil (1804)."""
    rows = []
    for reg, label in ABLATION_STEPS:
        agg = _aggregate(load_per_query(ABLATION_STEM, reg))
        rows.append({"register": reg, "label": label, **agg})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-arm join: Arms 1/2A/2B from T06, Arm 2C from the runs
# ---------------------------------------------------------------------------

# T06 method id per arm (canonical headline rows).
T06_METHOD = {
    "arm1":  "T03_arm1_naive",
    "arm2a": "T04_summary_node_hybrid",
    "arm2b": "T05_pageindex",
}


def cross_arm_recall10() -> pd.DataFrame:
    """Per-document Recall@10 for all four arms + a corpus-weighted row.

    Arms 1/2A/2B come from the T06 long frame (the single re-eval path); Arm 2C
    from the enriched+rerank runs. Columns: stem, stem_label, short, n,
    arm1, arm2a, arm2b, arm2c.
    """
    long = lr.load_long()
    ps = lr.per_stem_wide(long, ["R@10"])
    t06 = {(r["stem"], r["method"]): float(r["R@10"]) for _, r in ps.iterrows()}

    c2 = arm2c_per_pdf().set_index("stem")
    rows = []
    for stem, label, short in DOCS:
        n = int(c2.loc[stem, "n"])
        rows.append({
            "stem": stem, "stem_label": label, "short": short, "n": n,
            "arm1":  t06.get((stem, T06_METHOD["arm1"])),
            "arm2a": t06.get((stem, T06_METHOD["arm2a"])),
            "arm2b": t06.get((stem, T06_METHOD["arm2b"])),
            "arm2c": float(c2.loc[stem, "R@10"]),
        })
    df = pd.DataFrame(rows)
    w = df["n"]
    wt = {"stem": "corpus", "stem_label": "Corpus-weighted", "short": "Corpus",
          "n": int(w.sum())}
    for a in ("arm1", "arm2a", "arm2b", "arm2c"):
        wt[a] = (df[a] * w).sum() / w.sum()
    return pd.concat([df, pd.DataFrame([wt])], ignore_index=True)


# ---------------------------------------------------------------------------
# Full metric battery from the raw ranked lists (q*.json) -- so Arm 2C can slot
# into the cross-arm T06-frame builders that need R@5/@20, nDCG, etc. that the
# per_query.csv summary does not carry.
# ---------------------------------------------------------------------------

# Synthetic method id for the cross-arm frames (Arm 2C is not a T06 method).
ARM2C_METHOD = "ARM2C_agentic"

# The k cutoffs the cross-arm recall artefacts use (thesis_style.RECALL_K_CURVE).
CURVE_K = [5, 10, 20, 100]

# clean metric name -> T07 long-frame key (the keys thesis_style.resolve_metric
# expects); arm2c emits exactly these so get_variant/per_stem_wide resolve it.
_LONG_METRIC_KEYS = {
    "R@5": "T1/R@5", "R@10": "T1/R@10", "R@20": "T1/R@20", "R@100": "T1/R@100",
    "Hit@10": "T2/P1/HitRate@10", "MRR@10": "T2/P2/MRR@10",
    "nDCG@10": "T2/P2/NDCG@10", "P@10": "T2/P1/Precision@10",
}


def _as_int_list(x) -> list[int]:
    out = []
    for v in (x or []):
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            pass
    return out


def _iter_queries(stem: str, register: str = "enriched_rerank"):
    """Yield (gold_set, ranked_list) per query from the run's q*.json files."""
    for f in glob.glob(str(_run_dir(stem, register) / "q*.json")):
        d = json.load(open(f, encoding="utf-8"))
        gold = set(_as_int_list(d.get("gold_bsard_ids")))
        ranked = _as_int_list(d.get("ranked_bsard_ids"))
        if gold:
            yield gold, ranked


def _query_metrics(gold: set[int], ranked: list[int]) -> dict:
    """Binary IR metrics for one query from its ranked list + gold set."""
    g = gold
    m: dict = {}
    for k in CURVE_K:
        topk = ranked[:k]
        m[f"R@{k}"] = len(g.intersection(topk)) / len(g)
    top10 = ranked[:10]
    hit = g.intersection(top10)
    m["Hit@10"] = 1.0 if hit else 0.0
    m["P@10"] = len(hit) / 10.0
    # MRR@10
    rr = 0.0
    for i, a in enumerate(top10):
        if a in g:
            rr = 1.0 / (i + 1)
            break
    m["MRR@10"] = rr
    # nDCG@10 (standard: discount 1/log2(rank+1), rank from 1)
    dcg = sum((1.0 / math.log2(i + 2)) for i, a in enumerate(top10) if a in g)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(g), 10)))
    m["nDCG@10"] = (dcg / idcg) if idcg > 0 else 0.0
    return m


def _per_pdf_metric_means(register: str = "enriched_rerank") -> dict[str, dict]:
    """{stem: {clean_metric: mean, 'n': n}} computed from q*.json."""
    out: dict[str, dict] = {}
    for stem, _lab, _short in DOCS:
        rows = [_query_metrics(g, r) for g, r in _iter_queries(stem, register)]
        n = len(rows)
        agg = {clean: sum(r[clean] for r in rows) / n
               for clean in rows[0]} if n else {}
        agg["n"] = n
        out[stem] = agg
    return out


def arm2c_long_rows(register: str = "enriched_rerank") -> pd.DataFrame:
    """Arm-2C rows in the T06 cross_arm_long schema.

    Columns: stem, stem_label, method, n_method_q, n_gt_q, metric, value.
    Carries the binary battery (R@5/@10/@20/@100, Hit@10, MRR@10, nDCG@10, P@10)
    plus E/ proxies: because the run's gold is already clipped to each PDF's
    reachable articles, binary recall *is* drift-aware, so E/recall@k == R@k and
    E/n_evaluable == n_gt_q (documented; not a separate re-evaluation).
    """
    means = _per_pdf_metric_means(register)
    recs: list[dict] = []
    for stem, label, _short in DOCS:
        agg = means[stem]
        n = agg["n"]
        base = {"stem": stem, "stem_label": label, "method": ARM2C_METHOD,
                "n_method_q": n, "n_gt_q": n}
        for clean, key in _LONG_METRIC_KEYS.items():
            recs.append({**base, "metric": key, "value": agg[clean]})
        # E/ proxies (gold already reachable-clipped).
        recs.append({**base, "metric": "E/recall@10", "value": agg["R@10"]})
        recs.append({**base, "metric": "E/recall@100", "value": agg["R@100"]})
        recs.append({**base, "metric": "E/n_evaluable", "value": n})
    return pd.DataFrame(recs)


def arm2c_per_query_rows(register: str = "enriched_rerank") -> pd.DataFrame:
    """Arm-2C rows in the per_query_recall schema (stem, qid, method, k, recall)."""
    recs: list[dict] = []
    for stem, _label, _short in DOCS:
        for f in glob.glob(str(_run_dir(stem, register) / "q*.json")):
            d = json.load(open(f, encoding="utf-8"))
            gold = set(_as_int_list(d.get("gold_bsard_ids")))
            ranked = _as_int_list(d.get("ranked_bsard_ids"))
            if not gold:
                continue
            qid = d.get("query_id")
            for k in CURVE_K:
                recs.append({"stem": stem, "qid": qid, "method": ARM2C_METHOD,
                             "k": k,
                             "recall": len(gold.intersection(ranked[:k])) / len(gold)})
    return pd.DataFrame(recs)


def arm2c_cardinality(register: str = "enriched_rerank") -> dict[str, tuple[float, int]]:
    """{'single': (recall@10, n), 'multi': (recall@10, n)} pooled over PDFs."""
    single, multi = [], []
    for stem, _label, _short in DOCS:
        for g, r in _iter_queries(stem, register):
            rec = len(g.intersection(r[:10])) / len(g)
            (single if len(g) == 1 else multi).append(rec)
    return {
        "single": (sum(single) / len(single) if single else float("nan"), len(single)),
        "multi": (sum(multi) / len(multi) if multi else float("nan"), len(multi)),
    }


def arm2c_precision_ceiling(config: str = "simple") -> pd.DataFrame:
    """Per-question article precision@10 + achievable ceiling for the agentic arm,
    in the schema of the cross-arm precision-ceiling figure. Columns: stem,
    stem_label, achieved, ceiling, where
        achieved = n_gold@10 / distinct_articles@10
        ceiling  = min(n_gold, distinct@10) / distinct@10.
    config='simple' reads the shipped run's ranked lists (distinct@10 exact);
    config='cnr' uses e5_corr (n_gold@10 = e5_corr*n_gold) with distinct@10 = 10
    (the agentic top-10 is 10 distinct articles; the e5 list itself was not saved).
    """
    rows = []
    if config == "cnr":
        e2dir = ARM2C_DIR / "runs" / "_corrective_e2"
        for stem, label, _short in DOCS:
            df = pd.read_csv(e2dir / f"{stem}_e2_text.csv")
            for _, r in df.iterrows():
                ng = int(r["n_gold"])
                if ng == 0:
                    continue
                gold10 = round(float(r["e5_corr"]) * ng)
                rows.append({"stem": stem, "stem_label": label,
                             "achieved": gold10 / 10.0,
                             "ceiling": min(ng, 10) / 10.0})
        return pd.DataFrame(rows)
    for stem, label, _short in DOCS:
        for g, ranked in _iter_queries(stem):
            top10 = ranked[:10]
            distinct = len(set(top10))
            if distinct == 0:
                continue
            gold10 = len(g.intersection(top10))
            rows.append({"stem": stem, "stem_label": label,
                         "achieved": gold10 / distinct,
                         "ceiling": min(len(g), distinct) / distinct})
    return pd.DataFrame(rows)


def arm2c_cardinality_cnr() -> dict[str, tuple[float, int]]:
    """CNR analogue of arm2c_cardinality: {'single': (e5 recall@10, n), 'multi':
    (recall@10, n)} pooled over PDFs, split by gold cardinality, from the
    controlled E2 per-query output (e5_corr)."""
    e2dir = ARM2C_DIR / "runs" / "_corrective_e2"
    single, multi = [], []
    for stem, _label, _short in DOCS:
        df = pd.read_csv(e2dir / f"{stem}_e2_text.csv")
        for _, r in df.iterrows():
            (single if int(r["n_gold"]) == 1 else multi).append(float(r["e5_corr"]))
    return {
        "single": (sum(single) / len(single) if single else float("nan"), len(single)),
        "multi": (sum(multi) / len(multi) if multi else float("nan"), len(multi)),
    }


def arm2c_reach_decomp(register: str = "enriched_rerank") -> pd.DataFrame:
    """Per-PDF question-level recall decomposition (the Arm-2B navigated-vs-padded
    analogue). All means are question-weighted, so the tiers nest:

        selected_recall  <=  recall@10  <=  padded_recall@100  <=  reached_recall

    - selected_recall    : recall over the agent's explicit picks (selected_bsard_ids)
                           -- the "exposed head" analogue.
    - recall@10          : the shipped headline (final, re-ranked top-10).
    - padded_recall@100  : recall over the full padded top-100.
    - reached_recall     : (gold_HIT + gold_SEEN_NOT_SELECTED) / n_gold -- the
                           navigation coverage ceiling (what descent ever reached);
                           the residual to 1.0 is NOT_REACHED + ORPHAN.
    """
    rows: list[dict] = []
    for stem, label, short in DOCS:
        pq = load_per_query(stem, register)
        sel_recall: dict[int, float] = {}
        for f in glob.glob(str(_run_dir(stem, register) / "q*.json")):
            d = json.load(open(f, encoding="utf-8"))
            gold = set(_as_int_list(d.get("gold_bsard_ids")))
            sel = set(_as_int_list(d.get("selected_bsard_ids")))
            if gold:
                sel_recall[int(d["query_id"])] = len(gold & sel) / len(gold)
        valid = pq[pq["n_gold"] > 0].copy()
        reached = ((valid["gold_HIT"] + valid["gold_SEEN_NOT_SELECTED"])
                   / valid["n_gold"])
        selected = valid["qid"].map(lambda q: sel_recall.get(int(q)))
        rows.append({
            "stem": stem, "stem_label": label, "short": short, "n": len(valid),
            "selected_recall": float(selected.mean()),
            "recall@10": float(valid["recall@10"].mean()),
            "padded_recall@100": float(valid["recall@100"].mean()),
            "reached_recall": float(reached.mean()),
        })
    return pd.DataFrame(rows)


def arm2c_ablation_curve(stem: str = ABLATION_STEM) -> dict[str, dict]:
    """{register: {'label', 'n', 'R@5', 'R@10', 'R@20', 'R@100'}} for the three
    ablation steps on one PDF (default Code Civil), from each run's q*.json."""
    out: dict[str, dict] = {}
    for reg, label in ABLATION_STEPS:
        rows = [_query_metrics(g, r) for g, r in _iter_queries(stem, reg)]
        n = len(rows)
        out[reg] = {"label": label, "n": n,
                    **{f"R@{k}": sum(x[f"R@{k}"] for x in rows) / n for k in CURVE_K}}
    return out


# ---------------------------------------------------------------------------
# Failure decomposition (the "why does it win/lose vs PageIndex" diagnostic).
# Maps every gold article to its tree location and classifies its fate per
# question, so the HIT share equals recall@10 exactly and the loss anatomy
# (where in the descent it broke) is on the same axis as the PageIndex tick.
# ---------------------------------------------------------------------------

_BSARD_RE = re.compile(r"bsard_(\d+)")
_ORPHAN_RE = re.compile(r"non rattach|unfiled|ungroup", re.I)
# fate classes, success-first then by descent level the miss occurred at.
FATE_CLASSES = ["hit", "misrank", "fine", "coarse", "orphan"]


def _deep_tree(stem: str) -> dict:
    p = ARM2C_DIR / "bundles" / stem / "deep_tree.json"
    if not p.exists():
        p = ARM2C_DIR / "data" / stem / "deep_tree.json"
    return json.load(open(p, encoding="utf-8"))["tree"]


def _collapse_root(node: dict) -> dict:
    while len(node.get("sub_nodes", [])) == 1:
        node = node["sub_nodes"][0]
    return node


def _tree_index(stem: str) -> dict:
    """bsard -> (section_id, top_branch_id); plus which top branches are the
    orphan catch-all. Top branches = children of the first multi-way node."""
    tops = _collapse_root(_deep_tree(stem)).get("sub_nodes", [])
    b_sec, b_top, top_orphan = {}, {}, {}

    def walk(n, top_id, stack):
        m = _BSARD_RE.search(n.get("node_id", ""))
        if m:
            sec = stack[-1] if stack else n["node_id"]
            b = int(m.group(1))
            b_sec[b], b_top[b] = sec, top_id
            return
        for c in n.get("sub_nodes", []):
            walk(c, top_id, stack + [n["node_id"]])

    for tb in tops:
        top_orphan[tb["node_id"]] = bool(_ORPHAN_RE.search(tb.get("title", "")))
        walk(tb, tb["node_id"], [])
    return {"b_sec": b_sec, "b_top": b_top, "top_orphan": top_orphan,
            "sec_top": {s: t for s, t in
                        ((b_sec[g], b_top[g]) for g in b_sec)},
            "n_tops": len(tops)}


def arm2c_failure_decomp(register: str = "enriched_rerank") -> pd.DataFrame:
    """Per-PDF question-weighted gold-fate decomposition + the 2C/2B/2A R@10.

    For each question every gold article is classified, success-first, into the
    descent level at which it was won or lost:
      hit      -- retrieved in the final top-10 (this share == Arm-2C recall@10);
      misrank  -- reached (its section was expanded) but ranked out of the top-10;
      fine     -- the right top-branch was entered, its section never expanded;
      coarse   -- the top-branch holding it was never entered;
      orphan   -- it sits in the broken-hierarchy catch-all branch.
    Columns: stem, stem_label, short, n, hit, misrank, fine, coarse, orphan,
    arm2b, arm2a  (last two = PageIndex / embedding recall@10 from T06).
    """
    cross = cross_arm_recall10().set_index("stem")
    rows = []
    for stem, label, short in DOCS:
        idx = _tree_index(stem)
        b_sec, b_top, top_orphan = idx["b_sec"], idx["b_top"], idx["top_orphan"]
        sec_top = idx["sec_top"]
        acc = {k: 0.0 for k in FATE_CLASSES}
        nq = 0
        for f in glob.glob(str(_run_dir(stem, register) / "q*.json")):
            d = json.load(open(f, encoding="utf-8"))
            gold = [int(x) for x in d.get("gold_bsard_ids", [])]
            if not gold:
                continue
            nq += 1
            visited = {s.get("node_id") for s in d.get("steps", [])}
            top10 = set(int(x) for x in d.get("ranked_bsard_ids", [])[:10])
            per = {k: 0 for k in FATE_CLASSES}
            for g in gold:
                top, sec = b_top.get(g), b_sec.get(g)
                if g in top10:
                    per["hit"] += 1
                elif top is None or top_orphan.get(top):
                    per["orphan"] += 1
                elif sec in visited:
                    per["misrank"] += 1
                elif any(sec_top.get(v) == top for v in visited):
                    per["fine"] += 1
                else:
                    per["coarse"] += 1
            for k in FATE_CLASSES:
                acc[k] += per[k] / len(gold)
        row = {"stem": stem, "stem_label": label, "short": short, "n": nq,
               **{k: acc[k] / nq for k in FATE_CLASSES},
               "arm2b": float(cross.loc[stem, "arm2b"]),
               "arm2a": float(cross.loc[stem, "arm2a"])}
        rows.append(row)
    return pd.DataFrame(rows)


def arm2c_failure_decomp_cnr() -> pd.DataFrame:
    """Per-PDF CNR gold-fate decomposition (reach-first) + the R@10 ladder.

    CNR = Corrective Navigation and Re-Ranking. For each question every gold
    article is classified from the corrective navigation (``corr_visit`` in the
    E2 nav.json) + the tree, success/reach-first:
      hit      -- in the final e5 top-10 (this share == CNR recall@10);
      misrank  -- reached (its section was expanded by the corrective descent)
                  but ranked out of the top-10  (reached - hit);
      fine     -- the right top-branch was entered, its section never expanded;
      coarse   -- the top-branch holding it was never entered;
      orphan   -- broken-hierarchy catch-all.
    'reached' comes from corr_visit + the tree; 'hit' is the per-query e5
    recall@10 (e2_text ``e5_corr``). The success-first vs reach-first nuance (a
    padded top-10 hit beyond the reached pool) is negligible: hit <= reached holds
    (hit is clamped to reached), so the bars sum to 1 and the green segment equals
    CNR recall@10 to displayed precision. The R@10 ladder columns are all from the
    controlled E2 run except the two reference arms (T06): arm2b, arm2a (R@10),
    arm2c_simple (8B/single), arm2c_cnr (e5/corrective). Columns: stem,
    stem_label, short, n, hit, misrank, fine, coarse, orphan, arm2b, arm2a,
    arm2c_simple, arm2c_cnr.
    """
    cross = cross_arm_recall10().set_index("stem")
    ea = arm2c_e2_all().set_index("stem")
    e2dir = ARM2C_DIR / "runs" / "_corrective_e2"
    rows = []
    for stem, label, short in DOCS:
        idx = _tree_index(stem)
        b_sec, b_top, top_orphan, sec_top = (idx["b_sec"], idx["b_top"],
                                             idx["top_orphan"], idx["sec_top"])
        e2 = {int(r["qid"]): r for _, r in
              pd.read_csv(e2dir / f"{stem}_e2_text.csv").iterrows()}
        nav = {int(r["qid"]): r for r in
               json.load(open(e2dir / f"{stem}_nav.json", encoding="utf-8"))["rows"]}
        acc = {k: 0.0 for k in FATE_CLASSES}
        nq = 0
        for qid, r in nav.items():
            gold = [int(x) for x in r.get("gold_bsard_ids", [])]
            if qid not in e2 or not gold:
                continue
            nq += 1
            visited = set(r["corr_visit"])
            per = {"fine": 0, "coarse": 0, "orphan": 0}
            reached = 0
            for g in gold:
                top, sec = b_top.get(g), b_sec.get(g)
                if sec in visited:
                    reached += 1
                elif top is None or top_orphan.get(top):
                    per["orphan"] += 1
                elif any(sec_top.get(v) == top for v in visited):
                    per["fine"] += 1
                else:
                    per["coarse"] += 1
            ng = len(gold)
            reach_frac = reached / ng
            hit = min(float(e2[qid]["e5_corr"]), reach_frac)
            acc["hit"] += hit
            acc["misrank"] += reach_frac - hit
            for k in per:
                acc[k] += per[k] / ng
        rows.append({
            "stem": stem, "stem_label": label, "short": short, "n": nq,
            **{k: acc[k] / nq for k in FATE_CLASSES},
            "arm2b": float(cross.loc[stem, "arm2b"]),
            "arm2a": float(cross.loc[stem, "arm2a"]),
            "arm2c_simple": float(ea.loc[stem, "r10_8B_single"]),
            "arm2c_cnr": float(ea.loc[stem, "r10_e5_corr"]),
        })
    return pd.DataFrame(rows)


def _tree_reach_index(stem: str) -> dict:
    """bsard -> dict(dec, sec_size, orphan): #real (multi-way) decisions to reach
    the article, its parent section size, and whether it sits in the orphan
    catch-all branch. Drives the reachability-barrier diagnostic."""
    root = _collapse_root(_deep_tree(stem))
    out: dict[int, dict] = {}

    def walk(n, dec, sizes, orph):
        m = _BSARD_RE.search(n.get("node_id", ""))
        if m:
            out[int(m.group(1))] = {"dec": dec, "sec_size": sizes[-1] if sizes else 0,
                                    "orphan": orph}
            return
        k = len(n.get("sub_nodes", []))
        child_orph = orph or bool(_ORPHAN_RE.search(n.get("title", "")))
        for c in n.get("sub_nodes", []):
            walk(c, dec + (1 if k > 1 else 0), sizes + [k], child_orph)

    top = root.get("sub_nodes", [])
    for c in top:
        walk(c, 1, [len(top)], bool(_ORPHAN_RE.search(c.get("title", ""))))
    return out


def arm2c_reachability_barriers(register: str = "enriched_rerank",
                                oversized: int = 12) -> pd.DataFrame:
    """Per-PDF structural barriers to reaching gold (distinct gold articles).

    Separates HARD blocks (orphan catch-all, oversized non-orphan section) from
    the SOFT depth gradient (number of correct sequential branch choices needed).
    Columns: stem, stem_label, short, n_gold, orphan_pct, oversized_pct,
    dec1, dec2, dec3, dec4plus (fractions of gold), mean_dec.
    """
    rows = []
    for stem, label, short in DOCS:
        idx = _tree_reach_index(stem)
        gold = set()
        for f in glob.glob(str(_run_dir(stem, register) / "q*.json")):
            d = json.load(open(f, encoding="utf-8"))
            gold.update(int(x) for x in d.get("gold_bsard_ids", []))
        g = [b for b in gold if b in idx]
        n = len(g)
        orph = sum(idx[b]["orphan"] for b in g)
        over = sum((not idx[b]["orphan"]) and idx[b]["sec_size"] > oversized for b in g)
        decs = [idx[b]["dec"] for b in g]
        b1 = sum(d <= 1 for d in decs); b2 = sum(d == 2 for d in decs)
        b3 = sum(d == 3 for d in decs); b4 = sum(d >= 4 for d in decs)
        rows.append({
            "stem": stem, "stem_label": label, "short": short, "n_gold": n,
            "orphan_pct": orph / n, "oversized_pct": over / n,
            "dec1": b1 / n, "dec2": b2 / n, "dec3": b3 / n, "dec4plus": b4 / n,
            "mean_dec": sum(decs) / n,
        })
    return pd.DataFrame(rows)


def pure_comparison(register: str = "enriched_rerank") -> pd.DataFrame:
    """Padding-free 2B-vs-2C comparison on the two navigator capabilities.

    Strips the deterministic padding both arms lean on and compares what each
    navigator actually reaches and selects (question-weighted recall):
      - reach   : 2C reached_recall (gold whose section was expanded) vs
                  2B nav_recall (gold in the navigated candidate set).
      - select  : 2C selected_recall (gold among selected_bsard_ids) vs
                  2B exposed_recall (gold among the LLM's score>0 picks).
    Also carries CNR's two analogues from the controlled E2 run -- cc_reach =
    corrective reach (reach_corr), cc_select = the e5 ranking head over the
    reached pool (r10_e5_corr; padding-free, since the corrective pool >= 10). CNR
    has no agent-selection step, so its "selection" IS the e5 top-10. Plus each
    arm's padded recall@10 for reference. Arm-2B columns come from its deep-dive
    per-PDF summary (the single re-eval path). Columns: stem, stem_label, short, n,
    c_reach, b_reach, cc_reach, c_select, b_select, cc_select, c_padded10,
    b_padded10.
    """
    b_path = ts.T06_DATA_DIR / "tables" / "arm2b_per_pdf_summary.csv"
    b = pd.read_csv(b_path)[["stem", "nav_recall", "exposed_recall", "recall@10"]]
    c = arm2c_reach_decomp(register)
    m = c.merge(b, on="stem", how="left")
    ea = arm2c_e2_all().set_index("stem")
    return pd.DataFrame({
        "stem": m["stem"], "stem_label": m["stem_label"], "short": m["short"],
        "n": m["n"],
        "c_reach": m["reached_recall"], "b_reach": m["nav_recall"],
        "cc_reach": m["stem"].map(lambda s: float(ea.loc[s, "reach_corr"])),
        "c_select": m["selected_recall"], "b_select": m["exposed_recall"],
        "cc_select": m["stem"].map(lambda s: float(ea.loc[s, "r10_e5_corr"])),
        "c_padded10": m["recall@10_x"] if "recall@10_x" in m else m["recall@10"],
        "b_padded10": m["recall@10_y"] if "recall@10_y" in m else m["recall@10"],
    })


def arm2c_cost_row(register: str = "enriched_rerank") -> dict:
    """Cost block for costs_by_method: method, n_q, llm_calls_per_q, tokens_per_q,
    latency_ms_mean, latency_ms_p95. Latency = mean over queries of the summed
    per-step nav latency (excludes the single re-rank call; tokens not recorded).
    """
    llm_total = 0.0
    n_q = 0
    lat_per_q: list[float] = []
    for stem, _label, _short in DOCS:
        for f in glob.glob(str(_run_dir(stem, register) / "q*.json")):
            d = json.load(open(f, encoding="utf-8"))
            n_q += 1
            llm_total += float(d.get("llm_calls", 0) or 0)
            steps = d.get("steps") or []
            lat_per_q.append(sum(float(s.get("latency_ms", 0) or 0) for s in steps))
    lat = pd.Series(lat_per_q)
    return {
        "method": ARM2C_METHOD, "n_q": n_q,
        "llm_calls_per_q": llm_total / n_q if n_q else float("nan"),
        "tokens_per_q": float("nan"),
        "latency_ms_mean": float(lat.mean()) if n_q else float("nan"),
        "latency_ms_p95": float(lat.quantile(0.95)) if n_q else float("nan"),
    }


# ---------------------------------------------------------------------------
# Corrective-loop (CRAG x ReAct) -> embedding-rerank synthesis (E2).
# ---------------------------------------------------------------------------


def arm2c_e2_summary(stem: str = ABLATION_STEM) -> dict:
    """E2 (corrective nav + e5 rerank) aggregate for one PDF: the four R@10 configs
    + reach. Prefers the real per-query phase-2 output
    (runs/_corrective_e2/<stem>_e2_text.csv); for Code Civil falls back to the
    transcribed key-value summary (civil_e2_summary.csv) if the CSV isn't present.
    Keys: r10_{8B,e5}_{single,corr}, reach_{single,corr}, n.
    """
    e2_dir = ARM2C_DIR / "runs" / "_corrective_e2"
    per_q = e2_dir / f"{stem}_e2_text.csv"
    if per_q.exists():
        df = pd.read_csv(per_q)
        return {
            "n": len(df),
            "reach_single": df["reach_single"].mean(), "reach_corr": df["reach_corr"].mean(),
            "r10_8B_single": df["r8_single"].mean(), "r10_8B_corr": df["r8_corr"].mean(),
            "r10_e5_single": df["e5_single"].mean(), "r10_e5_corr": df["e5_corr"].mean(),
        }
    if stem == ABLATION_STEM:
        kv = pd.read_csv(e2_dir / "civil_e2_summary.csv")
        return {r["key"]: float(r["value"]) for _, r in kv.iterrows()}
    raise FileNotFoundError(f"no E2 phase-2 output for {stem}: {per_q}")


def arm2c_e2_all() -> pd.DataFrame:
    """E2 summary for every PDF that has a phase-2 output, + a corpus-weighted row.
    Columns: stem, stem_label, short, n, reach_single, reach_corr,
    r10_8B_single, r10_8B_corr, r10_e5_single, r10_e5_corr. Also joins the T06
    Arm-1/Arm-2A R@10 (arm1, arm2a) for reference."""
    cross = cross_arm_recall10().set_index("stem")
    rows = []
    for stem, label, short in DOCS:
        try:
            s = arm2c_e2_summary(stem)
        except FileNotFoundError:
            continue
        rows.append({"stem": stem, "stem_label": label, "short": short,
                     "arm1": float(cross.loc[stem, "arm1"]),
                     "arm2a": float(cross.loc[stem, "arm2a"]), **s})
    df = pd.DataFrame(rows)
    w = df["n"]
    wt = {"stem": "corpus", "stem_label": "Corpus-weighted", "short": "Corpus",
          "n": int(w.sum())}
    for c in df.columns:
        if c in ("stem", "stem_label", "short", "n"):
            continue
        wt[c] = (df[c] * w).sum() / w.sum()
    return pd.concat([df, pd.DataFrame([wt])], ignore_index=True)


def arm2c_e2_cost_perf() -> pd.DataFrame:
    """Per-PDF Simple-navigation vs CNR cost + outcome (question-weighted), + a
    corpus-weighted row -- the two-config navigation table.

    Cost = mean navigation LLM calls per query, from the E2 nav.json:
      - Simple = round-1 node calls (one 8B JSON call per visited node);
      - CNR    = total navigation calls (corrective rounds + the re-seed ranking
                 call). The e5 re-rank is not an LLM call and the 8B re-rank is
                 excluded, so this is *navigation* cost only.
    Outcome = padding-free reach and Recall@10 from the controlled E2 per-query
    output (arm2c_e2_all, question-weighted means):
      - Simple = 8B re-rank over the single-pass reach (shipped Arm-2C);
      - CNR    = e5 re-rank over the corrective-loop reach.
    Columns: stem, stem_label, short, n, llm_single, llm_cnr, reach_single,
    reach_cnr, r10_single, r10_cnr.
    """
    e2dir = ARM2C_DIR / "runs" / "_corrective_e2"
    ea = arm2c_e2_all().set_index("stem")
    rows = []
    for stem, label, short in DOCS:
        nav = json.load(open(e2dir / f"{stem}_nav.json", encoding="utf-8"))["rows"]
        n = len(nav)
        e = ea.loc[stem]
        rows.append({
            "stem": stem, "stem_label": label, "short": short, "n": n,
            "llm_single": sum(len(r["round1_visit"]) for r in nav) / n,
            "llm_cnr": sum(r["llm_calls"] for r in nav) / n,
            "reach_single": float(e["reach_single"]), "reach_cnr": float(e["reach_corr"]),
            "r10_single": float(e["r10_8B_single"]), "r10_cnr": float(e["r10_e5_corr"]),
        })
    df = pd.DataFrame(rows)
    w = df["n"]
    corp = {"stem": "corpus", "stem_label": "Corpus-weighted", "short": "Corpus",
            "n": int(w.sum())}
    for c in ("llm_single", "llm_cnr", "reach_single", "reach_cnr",
              "r10_single", "r10_cnr"):
        corp[c] = float((df[c] * w).sum() / w.sum())
    return pd.concat([df, pd.DataFrame([corp])], ignore_index=True)


def arm2c_e2_reach_padding() -> pd.DataFrame:
    """Per-PDF Simple-navigation vs CNR navigated-vs-padded tiers
    (question-weighted) + a corpus-weighted row -- the before/after of the
    navigated-vs-padded figure. For each config three nested tiers:
      r10       -- final Recall@10 (Simple = 8B/single, CNR = e5/corrective);
      reach     -- reached ceiling = padding-free navigation coverage;
      pad100    -- padded recall@100 (the full padded list) -- computed from the
                   E2 nav.json ranked lists (round1_ranked / corr_ranked); it is
                   rank-order-independent at depth 100, so the 8B-ordered list
                   gives the same @100 set as the e5 re-rank over the same pool.
    The point: padded@100 sits at ~the reach ceiling for both configs (padding
    fills up to reach, not beyond), and CNR pushes that ceiling -- and R@10 --
    right. Columns: stem, stem_label, short, n, reach_single, reach_cnr,
    r10_single, r10_cnr, pad100_single, pad100_cnr.
    """
    e2dir = ARM2C_DIR / "runs" / "_corrective_e2"
    ea = arm2c_e2_all().set_index("stem")
    rows = []
    for stem, label, short in DOCS:
        nav = json.load(open(e2dir / f"{stem}_nav.json", encoding="utf-8"))["rows"]
        n = len(nav)

        def pad100(key: str) -> float:
            t = 0.0
            for r in nav:
                g = {int(x) for x in r["gold_bsard_ids"]}
                if g:
                    t += len(g & {int(x) for x in r[key][:100]}) / len(g)
            return t / n

        e = ea.loc[stem]
        rows.append({
            "stem": stem, "stem_label": label, "short": short, "n": n,
            "reach_single": float(e["reach_single"]), "reach_cnr": float(e["reach_corr"]),
            "r10_single": float(e["r10_8B_single"]), "r10_cnr": float(e["r10_e5_corr"]),
            "pad100_single": pad100("round1_ranked"), "pad100_cnr": pad100("corr_ranked"),
        })
    df = pd.DataFrame(rows)
    w = df["n"]
    corp = {"stem": "corpus", "stem_label": "Corpus-weighted", "short": "Corpus",
            "n": int(w.sum())}
    for c in df.columns:
        if c in ("stem", "stem_label", "short", "n"):
            continue
        corp[c] = float((df[c] * w).sum() / w.sum())
    return pd.concat([df, pd.DataFrame([corp])], ignore_index=True)


def arm2c_cnr_cost_row() -> dict:
    """CNR cost block for the cross-arm cost table, pooled over all 725 questions:
    method, n_q, llm_calls_per_q (the corrective navigation's 8B calls, incl. the
    re-seed ranking call), tokens_per_q (not recorded -> nan), latency_ms_mean /
    latency_ms_p95 (the navigation ``secs`` from nav.json; navigation-only, like
    the simple-nav row -- the decoupled e5 re-rank is not included)."""
    e2dir = ARM2C_DIR / "runs" / "_corrective_e2"
    llm, lat = [], []
    for stem, _label, _short in DOCS:
        for r in json.load(open(e2dir / f"{stem}_nav.json", encoding="utf-8"))["rows"]:
            llm.append(float(r.get("llm_calls", 0) or 0))
            lat.append(float(r.get("secs", 0) or 0) * 1000.0)
    lat = pd.Series(lat)
    n = len(llm)
    return {"method": "ARM2C_CNR", "n_q": n,
            "llm_calls_per_q": sum(llm) / n if n else float("nan"),
            "tokens_per_q": float("nan"),
            "latency_ms_mean": float(lat.mean()) if n else float("nan"),
            "latency_ms_p95": float(lat.quantile(0.95)) if n else float("nan")}


# Full CNR battery, recovered by re-running the (deterministic) e5 rerank over the
# saved corrective pools and capturing the ordered list -- see
# corrective/e2_metrics_full.py, which writes runs/_corrective_e2/<stem>_e2_metrics.csv
# (per-query R@5/@10/@20/@100, Hit@10, MRR@10, nDCG@10, P@10 of the e5/corr list) and
# <stem>_e2_ranked.json (the ordered lists). R@10 reproduces the published e5_corr.
_CNR_BATTERY = ["R@5", "R@10", "R@20", "R@100", "Hit@10", "MRR@10", "nDCG@10", "P@10"]


def _cnr_per_query(stem: str) -> pd.DataFrame:
    p = ARM2C_DIR / "runs" / "_corrective_e2" / f"{stem}_e2_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"CNR full-battery file missing: {p}\n"
            f"  run: python corrective/e2_metrics_full.py")
    return pd.read_csv(p)


def _cnr_agg_df() -> pd.DataFrame:
    """Per-PDF + a 'corpus' row of the CNR battery means. Prefers the committed
    aggregate file (``arm2c_cnr_aggregates.csv``, produced from the Azure e5-rerank
    run — see corrective/e2_metrics_full.py); falls back to recomputing from the
    per-query ``<stem>_e2_metrics.csv`` if those are present locally. Columns:
    stem, n, R@5, R@10, R@20, R@100, Hit@10, MRR@10, nDCG@10, P@10."""
    agg = ARM2C_DIR / "runs" / "_corrective_e2" / "arm2c_cnr_aggregates.csv"
    if agg.exists():
        return pd.read_csv(agg)
    rows, tot, N = [], {k: 0.0 for k in _CNR_BATTERY}, 0
    for stem, _l, _s in DOCS:
        df = _cnr_per_query(stem)
        n = len(df)
        N += n
        mu = {k: float(df[k].mean()) for k in _CNR_BATTERY}
        for k in _CNR_BATTERY:
            tot[k] += mu[k] * n
        rows.append({"stem": stem, "n": n, **mu})
    rows.append({"stem": "corpus", "n": N, **{k: tot[k] / N for k in _CNR_BATTERY}})
    return pd.DataFrame(rows)


def arm2c_cnr_metrics() -> dict:
    """CNR (e5 + corrective) micro-averaged FULL battery over all 725 questions, for
    the cross-arm tables/curves. Recovered from the e5/corr ranked lists (the Azure
    e5-rerank run), so R@5/@20/MRR@10/nDCG@10 are now available (previously only
    R@10/R@100 were). R@10 reproduces the published 0.379. Gold is reachable-clipped
    so E/ == binary. Keys: R@5, R@10, R@20, R@100, Hit@10, MRR@10, nDCG@10, P@10,
    E/R@10, E/R@100, n.
    """
    c = _cnr_agg_df().set_index("stem").loc["corpus"]
    out = {k: float(c[k]) for k in _CNR_BATTERY}
    out["E/R@10"] = out["R@10"]
    out["E/R@100"] = out["R@100"]
    out["n"] = int(c["n"])
    return out


def arm2c_cnr_metrics_per_pdf() -> pd.DataFrame:
    """Per-PDF CNR battery (+ a corpus row), with display labels. Columns: stem,
    stem_label, short, n, R@5, R@10, R@20, R@100, Hit@10, MRR@10, nDCG@10, P@10."""
    df = _cnr_agg_df().set_index("stem")
    rows = []
    for stem, label, short in DOCS:
        r = df.loc[stem]
        rows.append({"stem": stem, "stem_label": label, "short": short,
                     "n": int(r["n"]), **{k: float(r[k]) for k in _CNR_BATTERY}})
    c = df.loc["corpus"]
    rows.append({"stem": "corpus", "stem_label": "Corpus-weighted", "short": "Corpus",
                 "n": int(c["n"]), **{k: float(c[k]) for k in _CNR_BATTERY}})
    return pd.DataFrame(rows)


def arm2c_e2_fate() -> pd.DataFrame:
    """Gold-article-weighted reach/hit decomposition, BEFORE (simple navigation)
    vs AFTER (CNR), for the before/after miss anatomy.

    Both columns come from the SAME controlled E2 per-query output
    (``runs/_corrective_e2/<stem>_e2_text.csv``) -- one navigation per query, so
    the contrast is confound-free (no run-to-run noise):
      - BEFORE = simple navigation = shipped Arm-2C = 8B re-rank over the
        single-pass reached pool (``reach_single`` / ``r8_single``).
      - AFTER  = CNR (Corrective Navigation and Re-Ranking) = e5 re-rank over the
        corrective-loop reached pool (``reach_corr`` / ``e5_corr``).
    For each config the gold articles split three ways (sum to 1):
      hit          -- in the final top-10            (== the config's recall@10)
      reached_miss -- reached but ranked out of top-10 (reached - hit)
      not_reached  -- the branch was never expanded (orphan catch-all folded in).
    All shares are gold-article-weighted (micro over n_gold), matching the
    miss-anatomy x-axis 'share of gold articles'. Columns: stem, stem_label,
    short, n, n_gold, hit_single, reached_single, hit_cnr, reached_cnr.

    Caveat: the E2 navigation is a *decoupled* re-run (high run-to-run variance),
    so the BEFORE bar differs slightly from the committed shipped
    ``arm2c_per_pdf`` miss anatomy -- the controlled within-E2 contrast is the
    scientifically clean one and is what this exposes.
    """
    e2_dir = ARM2C_DIR / "runs" / "_corrective_e2"
    rows = []
    for stem, label, short in DOCS:
        p = e2_dir / f"{stem}_e2_text.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        w = df["n_gold"]

        def gw(col: str) -> float:
            return float((df[col] * w).sum() / w.sum())

        rows.append({
            "stem": stem, "stem_label": label, "short": short,
            "n": len(df), "n_gold": float(w.sum()),
            "hit_single": gw("r8_single"), "reached_single": gw("reach_single"),
            "hit_cnr": gw("e5_corr"), "reached_cnr": gw("reach_corr"),
        })
    out = pd.DataFrame(rows)
    wg = out["n_gold"]
    corp = {"stem": "corpus", "stem_label": "Corpus-weighted", "short": "Corpus",
            "n": int(out["n"].sum()), "n_gold": float(wg.sum())}
    for c in ("hit_single", "reached_single", "hit_cnr", "reached_cnr"):
        corp[c] = float((out[c] * wg).sum() / wg.sum())
    return pd.concat([out, pd.DataFrame([corp])], ignore_index=True)


__all__ = [
    "ARM2C_DIR", "RUNS_DIR", "DOCS", "STEM_LABEL", "STEM_SHORT",
    "ABLATION_STEM", "ABLATION_STEPS", "MISS_CLASSES",
    "load_per_query", "arm2c_per_pdf", "arm2c_ablation",
    "T06_METHOD", "cross_arm_recall10",
    "ARM2C_METHOD", "CURVE_K",
    "arm2c_long_rows", "arm2c_per_query_rows", "arm2c_cardinality",
    "arm2c_reach_decomp", "arm2c_cardinality_cnr", "arm2c_precision_ceiling",
    "arm2c_ablation_curve", "arm2c_cost_row",
    "FATE_CLASSES", "arm2c_failure_decomp", "arm2c_failure_decomp_cnr",
    "pure_comparison",
    "arm2c_reachability_barriers",
    "arm2c_e2_summary", "arm2c_e2_all",
    "arm2c_e2_fate", "arm2c_e2_cost_perf", "arm2c_e2_reach_padding",
    "arm2c_cnr_metrics", "arm2c_cnr_cost_row",
]


if __name__ == "__main__":
    pd.set_option("display.width", 160, "display.max_columns", 30)
    print("=== cross-arm Recall@10 ===")
    print(cross_arm_recall10().to_string(index=False))
    print("\n=== arm2c per-pdf (enriched+rerank) ===")
    cols = ["stem_label", "n", "R@10", "R@10_pre", "R@100", "Hit@10",
            "nodes_per_q", "llm_calls_per_q", "selected_per_q",
            "gold_NOT_REACHED_frac", "gold_SEEN_NOT_SELECTED_frac"]
    print(arm2c_per_pdf()[cols].to_string(index=False))
    print("\n=== 1804 ablation ===")
    print(arm2c_ablation()[["label", "n", "R@10", "R@100", "Hit@10",
                            "empty_sel_q"]].to_string(index=False))
