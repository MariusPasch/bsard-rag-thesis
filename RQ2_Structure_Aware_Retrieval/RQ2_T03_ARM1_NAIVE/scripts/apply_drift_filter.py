"""Apply the drift filter to a PDF's existing cache without re-embedding.

Prerequisites:
  - ``drift_status.json`` already written for the doc (see compute_drift_status.py).
  - ``article_spans.json``, ``configs/<hash>/chunks.json``,
    ``configs/<hash>/faiss_meta.json``, and any
    ``results/<hash>/retrieval_pool/*.jsonl`` already exist.

The drift filter only affects per-chunk ``bsard_ids`` annotations — chunk text
and embeddings are independent of article spans (see
chunker.chunk_sliding_window). So this script does a surgical resync without
rebuilding FAISS / BM25 / re-embedding chunks:

  1. Filter ``article_spans.json`` to keep only IN_PDF bsard_ids and stamp
     the drift_filter_signature.
  2. For each config: recompute every chunk's ``bsard_ids`` via overlap with
     the filtered spans, write back ``chunks.json`` and sync the
     ``faiss_meta.json`` metadata copy.
  3. For each retrieval_pool jsonl: replace each candidate's ``bsard_ids``
     with the corresponding chunk's new bsard_ids.

Usage (from RQ2_T03_ARM1_NAIVE/, with the venv active):
    python scripts/apply_drift_filter.py --doc-id 1804_03_21_1804032153
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

DEFAULT_CACHE_ROOT = _REPO / "RQ2_T03_ARM1_NAIVE" / "data"
_HASH_LEN = 12


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(p: Path, payload: dict, *, compact: bool = False) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        if compact:
            json.dump(payload, f, ensure_ascii=False)
        else:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")


def _drift_signature(allowed: set[int]) -> str:
    blob = json.dumps(sorted(allowed), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:_HASH_LEN]


def _identify_bsard_ids(start: int, end: int, spans: list[dict]) -> list[int]:
    return [
        s["bsard_id"]
        for s in spans
        if s["start_char"] < end and s["end_char"] > start
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = ap.parse_args()

    doc_dir = args.cache_root / args.doc_id
    drift_path = doc_dir / "drift_status.json"
    spans_path = doc_dir / "article_spans.json"
    if not drift_path.exists():
        print(f"missing: {drift_path}\n"
              "Run compute_drift_status.py first.", file=sys.stderr)
        return 1
    if not spans_path.exists():
        print(f"missing: {spans_path}", file=sys.stderr)
        return 1

    drift = _read_json(drift_path)
    in_pdf_ids = {
        int(v["bsard_id"])
        for v in drift.get("verdicts", [])
        if v.get("verdict") == "IN_PDF"
    }
    signature = _drift_signature(in_pdf_ids)
    print(f"doc_id        = {args.doc_id}")
    print(f"IN_PDF bsard  = {len(in_pdf_ids)}")
    print(f"signature     = {signature}")

    # 1. article_spans.json — filter
    spans_doc = _read_json(spans_path)
    pre_n = len(spans_doc.get("spans", []))
    filtered_spans = [
        s for s in spans_doc.get("spans", [])
        if int(s["bsard_id"]) in in_pdf_ids
    ]
    spans_doc["spans"] = filtered_spans
    spans_doc["drift_filter_signature"] = signature
    spans_doc["computed_at"] = _now_utc()
    _write_json(spans_path, spans_doc)
    print(f"  article_spans.json   {pre_n:>5} -> {len(filtered_spans):<5} spans")

    # 2. each config: chunks.json + faiss_meta.json
    configs_dir = doc_dir / "configs"
    if not configs_dir.exists():
        print("  (no configs/ dir — done)")
        return 0

    for cfg_dir in sorted(p for p in configs_dir.iterdir() if p.is_dir()):
        chunks_path = cfg_dir / "chunks.json"
        if not chunks_path.exists():
            continue
        ck = _read_json(chunks_path)
        n_chunk_changed = 0
        for c in ck["chunks"]:
            new_ids = _identify_bsard_ids(
                int(c["start_char"]), int(c["end_char"]), filtered_spans
            )
            if list(c.get("bsard_ids", [])) != new_ids:
                c["bsard_ids"] = new_ids
                n_chunk_changed += 1
        _write_json(chunks_path, ck)
        print(f"  {cfg_dir.name}/chunks.json     "
              f"{n_chunk_changed:>4}/{ck['n_chunks']} chunks updated")

        chunks_by_id = {c["chunk_id"]: c for c in ck["chunks"]}

        fm_path = cfg_dir / "faiss_meta.json"
        if fm_path.exists():
            fm = _read_json(fm_path)
            n_meta_changed = 0
            for cid, meta in zip(fm["ids"], fm["metadata"]):
                new_ids = list(chunks_by_id[cid]["bsard_ids"])
                if meta.get("bsard_ids") != new_ids:
                    meta["bsard_ids"] = new_ids
                    n_meta_changed += 1
            _write_json(fm_path, fm, compact=True)
            print(f"  {cfg_dir.name}/faiss_meta.json "
                  f"{n_meta_changed:>4}/{len(fm['ids'])} entries updated")

        # 3. retrieval pools under results/<cfg>/retrieval_pool/*.jsonl
        pool_dir = doc_dir / "results" / cfg_dir.name / "retrieval_pool"
        if pool_dir.exists():
            for pool_path in sorted(pool_dir.glob("*.jsonl")):
                lines_out = []
                n_total = n_changed = 0
                with open(pool_path, encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        for cand in rec.get("candidates", []):
                            cid = cand["chunk_id"]
                            new_ids = list(chunks_by_id[cid]["bsard_ids"])
                            if cand.get("bsard_ids") != new_ids:
                                cand["bsard_ids"] = new_ids
                                n_changed += 1
                            n_total += 1
                        lines_out.append(
                            json.dumps(rec, ensure_ascii=False)
                        )
                with open(pool_path, "w", encoding="utf-8", newline="\n") as f:
                    for ln in lines_out:
                        f.write(ln + "\n")
                print(f"  {cfg_dir.name}/retrieval_pool/{pool_path.name}  "
                      f"{n_changed:>5}/{n_total} candidates updated")

    print("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
