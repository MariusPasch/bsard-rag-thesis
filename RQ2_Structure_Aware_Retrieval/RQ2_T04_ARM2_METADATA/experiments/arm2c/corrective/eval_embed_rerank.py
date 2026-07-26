"""Embedding-rerank vs LLM-rerank over the SHIPPED reached pool — the re-SELECT
lever, isolated. No Ollama: reconstructs each query's reached pool from the saved
single-pass traces (runs/arm2c_<stem>_enriched_rerank/q*.json) and re-ranks it with
Arm-2A's e5 encoder, then compares R@10 against the saved LLM-rerank over the SAME
pool. Runs over the full query set (only the e5 model is needed).

The question: the shipped run reaches R@100≈0.52 but ranks gold to only R@10≈0.33 —
can the embedding ranker convert that reached gold into the top-10?

    # real (needs sentence-transformers + the e5 model, GPU recommended):
    python corrective/eval_embed_rerank.py --stem 1804_03_21_1804032150
    # wiring check (no model): lexical mock encoder
    python corrective/eval_embed_rerank.py --stem 1804_03_21_1804032150 --mock --limit 30

Writes (additive): runs/_embed_rerank/<stem>_embed.{csv,md}
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ARM2C = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_ARM2C))
from navigator_tools import DeepTree                                   # noqa: E402
from embed_rerank import reached_pool_from_trace, embed_rerank_pool    # noqa: E402

OUT_DIR = _ARM2C / "runs" / "_embed_rerank"


# ── mock encoder (lexical) — validates wiring without the 2 GB model ──────────

class MockEncoder:
    """Deterministic char-3gram hashing encoder → normalised vectors. Cosine ≈
    lexical overlap; enough to exercise the pipeline (NOT a real semantic model)."""

    DIM = 512

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        t = (text or "").lower()
        for i in range(len(t) - 2):
            v[hash(t[i:i + 3]) % self.DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_query(self, query: str) -> np.ndarray:
        return self._vec(query)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts \
            else np.zeros((0, self.DIM), np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode_passages(texts)


# ── metrics ───────────────────────────────────────────────────────────────────

def _recall(gold: set[int], ranked: list[int], k: int) -> float:
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", required=True)
    ap.add_argument("--register", default="enriched_rerank")
    ap.add_argument("--text-field", choices=["text", "summary"], default="text",
                    help="embed raw French article text (Arm-2A article-unit) or the EN content summary")
    ap.add_argument("--limit", type=int, default=0, help="cap #queries (0 = all)")
    ap.add_argument("--mock", action="store_true", help="lexical mock encoder, no model")
    ap.add_argument("--device", default=None, help="torch device for e5 (e.g. cuda)")
    args = ap.parse_args()

    tree_path = _ARM2C / "bundles" / args.stem / "deep_tree.json"
    if not tree_path.exists():
        tree_path = _ARM2C / "data" / args.stem / "deep_tree.json"
    tree = DeepTree.load(tree_path)
    run_dir = _ARM2C / "runs" / f"arm2c_{args.stem}_{args.register}"
    qfiles = sorted(glob.glob(str(run_dir / "q*.json")))
    if args.limit:
        qfiles = qfiles[:args.limit]
    if not qfiles:
        print(f"no traces in {run_dir}"); return 1

    encoder = MockEncoder() if args.mock else None
    if encoder is None:
        from embed_rerank import E5Encoder
        print("loading e5 encoder (intfloat/multilingual-e5-large-instruct)…")
        encoder = E5Encoder(device=args.device)

    rows = []
    for f in qfiles:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        gold = set(int(x) for x in d.get("gold_bsard_ids", []))
        if not gold:
            continue
        ranked_llm = [int(x) for x in d.get("ranked_bsard_ids", [])]   # shipped LLM-rerank
        pool = reached_pool_from_trace(tree, d)
        pool_bsards = {tree.node(n).bsard_id for n in pool if tree.node(n)}
        reach = len(gold & pool_bsards) / len(gold)
        ranked_emb = embed_rerank_pool(d.get("query_text", ""), pool, tree, encoder,
                                       k=100, text_field=args.text_field)
        rows.append({
            "qid": d.get("query_id"), "n_gold": len(gold),
            "reach": round(reach, 3),
            "llm_r10": round(_recall(gold, ranked_llm, 10), 3),
            "emb_r10": round(_recall(gold, ranked_emb, 10), 3),
            "llm_r100": round(_recall(gold, ranked_llm, 100), 3),
            "emb_r100": round(_recall(gold, ranked_emb, 100), 3),
        })

    n = len(rows)
    M = lambda key: statistics.mean(r[key] for r in rows)
    summary = {k: M(k) for k in ("reach", "llm_r10", "emb_r10", "llm_r100", "emb_r100")}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "mock" if args.mock else args.text_field
    with (OUT_DIR / f"{args.stem}_embed_{tag}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    L = [f"# Embedding-rerank vs LLM-rerank — {args.stem} ({n} q, {tag})", "",
         "Same reached pool (reconstructed from the shipped single-pass traces); only",
         "the reranker changes. `reach` = gold present in the pool (padding-free ceiling).", "",
         f"- pool reach (ceiling) : {summary['reach']:.3f}",
         f"- R@10  LLM-rerank     : {summary['llm_r10']:.3f}   (shipped)",
         f"- R@10  embed-rerank   : {summary['emb_r10']:.3f}   (Δ {summary['emb_r10']-summary['llm_r10']:+.3f})",
         f"- R@100 LLM / embed    : {summary['llm_r100']:.3f} / {summary['emb_r100']:.3f}  (same pool → ~equal)",
         "",
         f"Conversion of reached gold into top-10: LLM {summary['llm_r10']/summary['reach']:.0%} "
         f"vs embed {summary['emb_r10']/summary['reach']:.0%} of the reachable ceiling."
         if summary['reach'] else ""]
    (OUT_DIR / f"{args.stem}_embed_{tag}.md").write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L))
    print(f"\n[wrote {OUT_DIR / f'{args.stem}_embed_{tag}.csv'} + .md]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
