"""Local re-derivation experiments for T05 PageIndex on PDF 1804.

Loads the post-fix per-query result bundles + the T05 tree + the T04 best-variant
ranking, then re-derives the T05 ranking under several ablations to see whether
cheap post-processing can lift R@K without spending more GPU time.

Run:
    python scripts/experiment_local_postprocess.py

All experiments are trace-based: no LLM calls, no Azure dependency.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--stem", default="1804_03_21_1804032150",
                 help="PDF stem (e.g. 1867_06_08_1867060850)")
_args, _ = _ap.parse_known_args()

ROOT = Path(__file__).resolve().parent.parent.parent
STEM = _args.stem
RUN = f"t04_{STEM}"

GT_PATH = ROOT / "RQ2_T07_EVALUATION/ground_truth/runs" / f"{RUN}.json"
T05_RESULTS_DIR = ROOT / f"RQ2_T05_ARM2_PAGEINDEX/data/{STEM}/results"
T05_TREE_PATH = ROOT / f"RQ2_T05_ARM2_PAGEINDEX/data/{STEM}/tree.json"

# T04 best variant for fusion: summary/node/hybrid/lv4 (winner at R@10 in the
# cross-arm table). One result file per query under results/<hash>/<run>.jsonl.
T04_BEST_HASH = "0a189183e751"   # full/node/hybrid/lv4 — close to summary winner
# Actually summary/node/hybrid/lv4 — we'll pick it by re-checking manifests at
# load time, since hashes are stable but variant labels are clearer.
T04_DIR = ROOT / f"RQ2_T04_ARM2_METADATA/data/{STEM}"

KS = (1, 5, 10, 20, 100)


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class T05Result:
    qid: str
    selected_law_ids: list[str]
    selected_chapter_ids: list[str]
    candidate_article_ids: list[str]          # LLM-picked, in selection order
    bsard_by_node: dict[str, int]             # node_id -> bsard_id (from ranked_items)
    article_scores: dict[str, float]          # node_id -> LLM score (1..3)
    empty_select_chapter_ids: list[str]       # chapters where article_selection returned []
    exit_reason: str


# ── Loaders ──────────────────────────────────────────────────────────────────


def load_gt() -> dict[str, set[int]]:
    raw = json.loads(GT_PATH.read_text(encoding="utf-8"))
    return {str(qid): {int(b) for b in bids} for qid, bids in raw.items()}


def load_tree() -> dict:
    return json.loads(T05_TREE_PATH.read_text(encoding="utf-8"))["tree"]


def find_node(root: dict, node_id: str) -> dict | None:
    if root.get("node_id") == node_id:
        return root
    for ch in root.get("sub_nodes", []):
        hit = find_node(ch, node_id)
        if hit is not None:
            return hit
    return None


def iter_leaves(node: dict):
    if node.get("article_id"):
        yield node
        return
    for ch in node.get("sub_nodes", []):
        yield from iter_leaves(ch)


def load_t05_results() -> dict[str, T05Result]:
    out: dict[str, T05Result] = {}
    for fp in sorted(T05_RESULTS_DIR.glob("q*.json")):
        d = json.loads(fp.read_text(encoding="utf-8"))
        qid = str(d["query_id"])
        summary = (d.get("trace") or {}).get("summary", {})
        steps = (d.get("trace") or {}).get("steps", [])

        # bsard_id mapping + LLM scores from ranked_items
        bsard_by_node = {}
        article_scores = {}
        for it in d.get("ranked_items", []):
            nid = it.get("id")
            bid = (it.get("metadata") or {}).get("bsard_id")
            if nid and bid is not None:
                bsard_by_node[nid] = int(bid)
            score = float(it.get("score") or 0.0)
            if score > 0 and nid:
                article_scores[nid] = score

        # Find chapters where article_selection returned an empty list.
        empty_chapters: list[str] = []
        for s in steps:
            name = str(s.get("step", ""))
            if not name.startswith("article_selection["):
                continue
            chapter_id = name[len("article_selection["):-1]
            parsed = s.get("parsed") or {}
            picked = parsed.get("selected_article_ids") if isinstance(parsed, dict) else None
            if not picked:
                empty_chapters.append(chapter_id)

        out[qid] = T05Result(
            qid=qid,
            selected_law_ids=summary.get("selected_law_ids", []) or [],
            selected_chapter_ids=summary.get("selected_chapter_ids", []) or [],
            candidate_article_ids=summary.get("candidate_article_ids", []) or [],
            bsard_by_node=bsard_by_node,
            article_scores=article_scores,
            empty_select_chapter_ids=empty_chapters,
            exit_reason=(d.get("cost") or {}).get("exit_reason", "?"),
        )
    return out


def pick_t04_best_hash() -> tuple[str, str]:
    """Pick the T04 variant with the highest R@10 on this PDF's GT.

    The cross-arm comparison showed the winner differs by PDF (1804 →
    summary/node/hybrid/lv4; 1867 → raw/article/hybrid/lv4), so picking
    statically would be misleading.
    """
    gt = load_gt()
    best_hash, best_label, best_r10 = None, None, -1.0
    for hash_dir in sorted((T04_DIR / "results").iterdir()):
        if not hash_dir.is_dir():
            continue
        jsonls = sorted(hash_dir.glob("*.jsonl"))
        if not jsonls:
            continue
        mf = json.loads(
            (T04_DIR / "configs" / hash_dir.name / "manifest.json").read_text(encoding="utf-8")
        )
        ch = mf.get("chunking", {})
        label = (f"T04 {ch.get('variant')}/{ch.get('unit')}/"
                 f"{'sparse' if ch.get('sparse_only') else 'hybrid'}/"
                 f"lv{ch.get('linker_version')}")
        qid_to_ranked = load_t04(hash_dir.name)
        m = score_arm(qid_to_ranked, gt)
        if m["R@10"] > best_r10:
            best_hash, best_label, best_r10 = hash_dir.name, label, m["R@10"]
    if best_hash is None:
        raise RuntimeError("No T04 variants found")
    return best_hash, best_label


def load_t04(hash_: str) -> dict[str, list[int]]:
    hash_dir = T04_DIR / "results" / hash_
    jsonls = sorted(hash_dir.glob("*.jsonl"))
    if not jsonls:
        raise RuntimeError(f"No jsonl in {hash_dir}")
    out = {}
    with jsonls[-1].open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            qid = str(d["query_id"])
            flat = []
            seen = set()
            for it in d.get("ranked_items", []):
                md = it.get("metadata", {}) or {}
                bids = md.get("bsard_ids") or md.get("bsard_id")
                if bids is None:
                    continue
                if isinstance(bids, list):
                    items = [int(b) for b in bids]
                else:
                    items = [int(bids)]
                for b in items:
                    if b in seen:
                        continue
                    seen.add(b)
                    flat.append(b)
            out[qid] = flat
    return out


# ── Metric ───────────────────────────────────────────────────────────────────


def recall_at_k(ranked: list[int], gt: set[int], k: int) -> float:
    if not gt:
        return 0.0
    return len(set(ranked[:k]) & gt) / len(gt)


def score_arm(qid_to_ranked: dict[str, list[int]], gt: dict[str, set[int]]) -> dict:
    common = set(qid_to_ranked.keys()) & set(gt.keys())
    rows = {f"R@{k}": [] for k in KS}
    for qid in common:
        ranked = qid_to_ranked[qid]
        gt_bids = gt[qid]
        for k in KS:
            rows[f"R@{k}"].append(recall_at_k(ranked, gt_bids, k))
    return {
        "n_q": len(common),
        **{k: float(np.mean(v)) if v else 0.0 for k, v in rows.items()},
    }


# ── Ranking constructors ─────────────────────────────────────────────────────


def dedupe_keep_order(seq):
    seen = set(); out = []
    for x in seq:
        if x in seen: continue
        seen.add(x); out.append(x)
    return out


def rank_llm_picks_only(r: T05Result, tie_breaker: str = "insertion") -> list[int]:
    """LLM-scored articles only, ordered by score desc with chosen tie breaker.

    tie_breaker:
      - 'insertion' (current navigator behaviour: stable sort by -score keeps
        candidate_article_ids order)
      - 'chapter_rank' (within a tied score, articles from earlier-selected
        chapters win; within a chapter, original order)
      - 'article_number_asc' (within a tied score, lower article_number wins)
    """
    # Build per-node (score, candidate_position, chapter_rank, article_number_int)
    chapter_rank = {ch: i for i, ch in enumerate(r.selected_chapter_ids)}
    # Need a way to know which chapter each article came from.
    # We have it implicitly: candidate_article_ids was built by iterating
    # selected_chapter_ids then article_selection results — chapters appear
    # in selection order, articles within them in their order.
    # We can't reconstruct exact assignment without re-walking article_selection
    # steps, but for tie-break purposes "first chapter contributes first" is a
    # good approximation: use the candidate_article_ids index.
    pos = {nid: i for i, nid in enumerate(r.candidate_article_ids)}
    nodes = [nid for nid in r.candidate_article_ids if nid in r.article_scores]

    def key(nid):
        s = -r.article_scores[nid]
        if tie_breaker == "insertion":
            return (s, pos[nid])
        if tie_breaker == "chapter_rank":
            # candidate_article_ids order already reflects chapter then article order
            # so this is the same as insertion for our trace shape.
            return (s, pos[nid])
        if tie_breaker == "article_number_asc":
            # We don't have article_number here — proxy via bsard_id ASC.
            bid = r.bsard_by_node.get(nid, 10**9)
            return (s, bid)
        raise ValueError(tie_breaker)

    nodes_sorted = sorted(nodes, key=key)
    return dedupe_keep_order(
        r.bsard_by_node[nid] for nid in nodes_sorted if nid in r.bsard_by_node
    )


def pad_with(scope: str, r: T05Result, tree_root: dict, existing: list[int],
             cap: int = 100) -> list[int]:
    """Append leaves from the given scope at the end of `existing`, deduped, up to `cap`."""
    seen = set(existing)
    out = list(existing)
    if scope == "none":
        return out[:cap]
    if scope == "selected_chapter":
        sources = r.selected_chapter_ids
    elif scope == "selected_law":
        sources = r.selected_law_ids or [n["node_id"] for n in [tree_root]
                                          if n.get("node_id", "").startswith("LAW_")]
    elif scope == "full_tree":
        sources = [tree_root["node_id"]]
    else:
        raise ValueError(scope)
    for src in sources:
        node = find_node(tree_root, src)
        if node is None:
            continue
        for leaf in iter_leaves(node):
            bid = leaf.get("bsard_id")
            if bid is None:
                continue
            bid = int(bid)
            if bid in seen:
                continue
            seen.add(bid)
            out.append(bid)
            if len(out) >= cap:
                return out
    return out


def rank_with_backfill(r: T05Result, tree_root: dict, backfill_n: int) -> list[int]:
    """LLM picks + backfill from EMPTY-selection chapters (up to backfill_n per chapter).

    Different from padding: this adds articles from chapters the LLM SELECTED
    but for which article_selection returned []. The idea: the chapter pick
    signal is still valid, even if the per-article pick failed.
    """
    llm = [r.bsard_by_node[n] for n in r.candidate_article_ids
           if n in r.article_scores and n in r.bsard_by_node]
    llm = dedupe_keep_order(llm)
    seen = set(llm)
    out = list(llm)
    for chapter_id in r.empty_select_chapter_ids:
        node = find_node(tree_root, chapter_id)
        if node is None:
            continue
        added = 0
        for leaf in iter_leaves(node):
            bid = leaf.get("bsard_id")
            if bid is None:
                continue
            bid = int(bid)
            if bid in seen:
                continue
            seen.add(bid)
            out.append(bid)
            added += 1
            if added >= backfill_n:
                break
    return out


def rrf_fuse(a: list[int], b: list[int], k_const: int = 60) -> list[int]:
    """Reciprocal rank fusion: score(d) = sum_arm 1/(k + rank_arm(d))."""
    scores: dict[int, float] = defaultdict(float)
    for rank, bid in enumerate(a):
        scores[bid] += 1.0 / (k_const + rank)
    for rank, bid in enumerate(b):
        scores[bid] += 1.0 / (k_const + rank)
    return [bid for bid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


# ── Experiments ──────────────────────────────────────────────────────────────


def exp_tie_break_diagnostic(t05: dict[str, T05Result]) -> None:
    """Print stats on LLM score distributions and tie sizes."""
    score_counts = defaultdict(int)
    rank1_ties = []   # for each query, how many articles share the top score?
    for r in t05.values():
        if not r.article_scores:
            continue
        scores = list(r.article_scores.values())
        for s in scores:
            score_counts[s] += 1
        top = max(scores)
        rank1_ties.append(sum(1 for s in scores if s == top))
    print("LLM-assigned score distribution (across all queries):")
    for s in sorted(score_counts.keys(), reverse=True):
        print(f"  score={s:>3.1f}  count={score_counts[s]}")
    print()
    if rank1_ties:
        print(
            f"Top-score tie size per query (n={len(rank1_ties)}): "
            f"mean={np.mean(rank1_ties):.2f}  median={int(np.median(rank1_ties))}  "
            f"max={max(rank1_ties)}  pct queries with ties (>=2): "
            f"{100*sum(1 for t in rank1_ties if t>=2)/len(rank1_ties):.1f}%"
        )


def run_experiments(t05: dict[str, T05Result], tree_root: dict,
                    gt: dict[str, set[int]], t04_best: dict[str, list[int]],
                    t04_label: str) -> pd.DataFrame:
    rows: list[dict] = []

    def add(label: str, ranker, requires_tree: bool = False):
        out = {}
        for qid, r in t05.items():
            ranking = ranker(r)
            out[qid] = ranking
        m = score_arm(out, gt)
        rows.append({"variant": label, **m})

    # --- 0. Current baseline: SCORE-SORTED LLM picks + selected-law padding ---
    # Matches what the navigator actually writes to disk (state.final_ranked is
    # `ranked.sort(key=lambda x: -x[1])`, stable on candidate_article_ids).
    def baseline(r: T05Result):
        nodes = [n for n in r.candidate_article_ids
                 if n in r.article_scores and n in r.bsard_by_node]
        nodes.sort(key=lambda n: -r.article_scores[n])   # stable
        llm_bids = dedupe_keep_order(r.bsard_by_node[n] for n in nodes)
        return pad_with("selected_law", r, tree_root, llm_bids, cap=100)
    add("0. baseline (sort by -score, then selected-law padding)", baseline)

    # --- 1. Tie-break variants (no padding so the effect is visible) ---------
    add("1a. LLM-only, insertion tie-break (current)",
        lambda r: dedupe_keep_order(
            r.bsard_by_node[n] for n in rank_llm_picks_only(r, "insertion").__iter__()
        ) if False else  # noqa  — fast-path inline
        [r.bsard_by_node[n] for n in r.candidate_article_ids
         if n in r.article_scores and n in r.bsard_by_node])

    # Simpler: just use the helper.
    rows.pop()
    def llm_only_insertion(r):
        # Sort LLM-picked nodes by -score, stable on candidate_article_ids order.
        nodes = [n for n in r.candidate_article_ids
                 if n in r.article_scores and n in r.bsard_by_node]
        nodes.sort(key=lambda n: -r.article_scores[n])   # stable
        return dedupe_keep_order(r.bsard_by_node[n] for n in nodes)
    add("1a. LLM-only, insertion tie-break (current)", llm_only_insertion)

    def llm_only_bsard_asc(r):
        # Within ties, prefer lower bsard_id (≈ lower article number).
        nodes = [n for n in r.candidate_article_ids
                 if n in r.article_scores and n in r.bsard_by_node]
        nodes.sort(key=lambda n: (-r.article_scores[n], r.bsard_by_node[n]))
        return dedupe_keep_order(r.bsard_by_node[n] for n in nodes)
    add("1b. LLM-only, bsard_id ASC tie-break", llm_only_bsard_asc)

    def llm_only_bsard_desc(r):
        # Within ties, prefer higher bsard_id (mostly newer articles).
        nodes = [n for n in r.candidate_article_ids
                 if n in r.article_scores and n in r.bsard_by_node]
        nodes.sort(key=lambda n: (-r.article_scores[n], -r.bsard_by_node[n]))
        return dedupe_keep_order(r.bsard_by_node[n] for n in nodes)
    add("1c. LLM-only, bsard_id DESC tie-break", llm_only_bsard_desc)

    # --- 2. Padding scope ablation (all use score-sorted LLM core) -----------
    def score_sorted_llm(r):
        nodes = [n for n in r.candidate_article_ids
                 if n in r.article_scores and n in r.bsard_by_node]
        nodes.sort(key=lambda n: -r.article_scores[n])
        return dedupe_keep_order(r.bsard_by_node[n] for n in nodes)

    for scope in ("none", "selected_chapter", "selected_law", "full_tree"):
        def make(scope=scope):
            def fn(r):
                return pad_with(scope, r, tree_root, score_sorted_llm(r), cap=100)
            return fn
        add(f"2. padding={scope}", make())

    # 2e — Two-tier: chapter first (concentrated), then law (fill the rest).
    # If chapter padding wins on its own, this should be at least as good and
    # also fill out R@100 like law padding does.
    def two_tier(r):
        core = score_sorted_llm(r)
        after_chapter = pad_with("selected_chapter", r, tree_root, core, cap=100)
        return pad_with("selected_law", r, tree_root, after_chapter, cap=100)
    add("2e. padding=selected_chapter THEN selected_law", two_tier)

    # --- 3. Article-selection backfill ---------------------------------------
    # Backfill empty-selection chapters BEFORE chapter+law padding, so the
    # backfilled articles get rank priority over the broad padding.
    for bn in (3, 6, 12):
        def make(bn=bn):
            def fn(r):
                # Score-sorted LLM picks at top
                core = score_sorted_llm(r)
                # Backfill from empty-selection chapters (first bn articles each)
                seen = set(core); out = list(core)
                for chapter_id in r.empty_select_chapter_ids:
                    node = find_node(tree_root, chapter_id)
                    if node is None: continue
                    added = 0
                    for leaf in iter_leaves(node):
                        bid = leaf.get("bsard_id")
                        if bid is None: continue
                        bid = int(bid)
                        if bid in seen: continue
                        seen.add(bid); out.append(bid); added += 1
                        if added >= bn: break
                # Then chapter padding (concentrated), then law padding (tail).
                out = pad_with("selected_chapter", r, tree_root, out, cap=100)
                return pad_with("selected_law", r, tree_root, out, cap=100)
            return fn
        add(f"3. backfill_n={bn} + chapter-then-law pad", make())

    # --- 4. T04 + T05 RRF fusion ---------------------------------------------
    # 4a — T04 only (sanity, should match the cross-arm table)
    rows.append({"variant": f"4a. {t04_label} (alone)", **score_arm(t04_best, gt)})

    # 4b — T05 baseline only (sanity)
    t05_baseline = {qid: baseline(r) for qid, r in t05.items()}
    rows.append({"variant": "4b. T05 baseline (alone)", **score_arm(t05_baseline, gt)})

    # 4c — RRF fusion
    fused = {}
    for qid in t05.keys() & t04_best.keys():
        fused[qid] = rrf_fuse(t05_baseline[qid], t04_best[qid])
    rows.append({"variant": f"4c. RRF(T05 baseline, {t04_label})",
                 **score_arm(fused, gt)})

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Loading GT … {GT_PATH}")
    gt = load_gt()
    print(f"  {len(gt)} questions, {sum(len(v) for v in gt.values())} GT bsard_ids\n")

    print(f"Loading T05 tree … {T05_TREE_PATH}")
    tree_root = load_tree()
    print(f"  root.node_id={tree_root['node_id']}\n")

    print(f"Loading T05 post-fix results … {T05_RESULTS_DIR}")
    t05 = load_t05_results()
    print(f"  {len(t05)} per-query results loaded\n")

    print("=== Tie-break diagnostic ===")
    exp_tie_break_diagnostic(t05)
    print()

    print(f"Loading T04 best variant …")
    t04_hash, t04_label = pick_t04_best_hash()
    print(f"  hash={t04_hash}  label={t04_label}")
    t04_best = load_t04(t04_hash)
    print(f"  {len(t04_best)} per-query rankings loaded\n")

    df = run_experiments(t05, tree_root, gt, t04_best, t04_label)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    for c in [f"R@{k}" for k in KS]:
        df[c] = df[c].round(4)
    print("=== Experiment results (PDF 1804, 252 questions) ===")
    print(df.to_string(index=False))

    # Persist for downstream plotting / report inclusion.
    out_dir = ROOT / f"RQ2_T05_ARM2_PAGEINDEX/data/{STEM}/experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "local_postprocess_experiments.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    base_row = df[df["variant"].str.startswith("0. ")].iloc[0]
    print()
    print(f"Baseline (variant 0) R@K:  R@1={base_row['R@1']:.4f}  "
          f"R@5={base_row['R@5']:.4f}  R@10={base_row['R@10']:.4f}  "
          f"R@20={base_row['R@20']:.4f}  R@100={base_row['R@100']:.4f}")
    print()
    print("Deltas vs baseline (R@10):")
    for _, row in df.iterrows():
        if row["variant"].startswith("0. "):
            continue
        d10 = row["R@10"] - float(base_row["R@10"])
        marker = "  <- WIN" if d10 > 0.005 else ("  (worse)" if d10 < -0.005 else "")
        print(f"  {row['variant']:<60}  R@10={row['R@10']:.4f}  "
              f"(delta {d10:+.4f}){marker}")


if __name__ == "__main__":
    main()
