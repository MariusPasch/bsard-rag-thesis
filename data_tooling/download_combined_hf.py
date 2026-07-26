"""Download one subset (or all) of the combined BSARD-RAG Hugging Face dataset.

Combined dataset layout (repo: mpaschalidis/bsard-rag-thesis-data, dataset):

    corpus/   rq1/   rq2/

By default each subset lands in the matching component's local data root, so the
components find their artefacts with no extra configuration:

    corpus -> <repo>/bsard2currentlawmatching/output
    rq1    -> <repo>/RQ1_Retrieval_Methods/output        (also used by rq3)
    rq2    -> <repo>/RQ2_Structure_Aware_Retrieval/data

Usage:
    python data_tooling/download_combined_hf.py                      # all subsets, default targets
    python data_tooling/download_combined_hf.py --subset corpus
    python data_tooling/download_combined_hf.py --subset rq2 --local-dir D:/data/rq2
    python data_tooling/download_combined_hf.py --repo OTHER/REPO --revision v1

Requires `huggingface_hub`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MONO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = os.environ.get("BSARD_HF_COMBINED_REPO", "mpaschalidis/bsard-rag-thesis-data")

# subset -> default local target (mirrors each component's data root).
DEFAULT_TARGETS = {
    "corpus": MONO_ROOT / "bsard2currentlawmatching" / "output",
    "rq1": MONO_ROOT / "RQ1_Retrieval_Methods" / "output",
    "rq2": MONO_ROOT / "RQ2_Structure_Aware_Retrieval" / "data",
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _reassemble_shards(local_dir: Path) -> None:
    """Rebuild any <name>.partNNN byte-shards into <name> per sharded_files.json,
    then remove the parts and the manifest. No-op if no manifest is present."""
    manifest = local_dir / "sharded_files.json"
    if not manifest.exists():
        return
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    for rel, info in spec.items():
        target = local_dir / rel
        parts = [local_dir / f"{rel}.part{i:03d}" for i in range(info["parts"])]
        if not all(p.exists() for p in parts):
            print(f"  WARNING: missing parts for {rel} — left as-is")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
        if "sha256" in info and _sha256(target) != info["sha256"]:
            print(f"  WARNING: checksum mismatch on {rel} — keeping parts")
            continue
        for p in parts:
            p.unlink()
        print(f"  reassembled {rel} ({info['size'] / 1e6:.1f} MB)")
    manifest.unlink()


def download_subset(repo: str, subset: str, revision: str | None, local_dir: Path,
                    allow_patterns: list[str] | None = None,
                    ignore_patterns: list[str] | None = None) -> None:
    """Download <repo>:<subset>/ into local_dir, flatten the prefix, reassemble shards.

    allow_patterns / ignore_patterns are SUBSET-RELATIVE globs (e.g. "embeddings/**");
    they are prefixed with "<subset>/" internally. Defaults to the whole subset.
    """
    import shutil
    local_dir.mkdir(parents=True, exist_ok=True)
    allow = [f"{subset}/{p}" for p in allow_patterns] if allow_patterns else [f"{subset}/*"]
    ignore = [f"{subset}/{p}" for p in ignore_patterns] if ignore_patterns else None
    # Pull only this subset into a scratch dir, then flatten "<subset>/" into local_dir.
    cache_root = local_dir / f".hf_{subset}"
    snapshot_download(repo_id=repo, repo_type="dataset", revision=revision,
                      local_dir=str(cache_root), allow_patterns=allow,
                      ignore_patterns=ignore)
    src = cache_root / subset
    if src.exists():
        for item in src.rglob("*"):
            if item.is_file():
                dest = local_dir / item.relative_to(src)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    dest.unlink()
                item.replace(dest)
    shutil.rmtree(cache_root, ignore_errors=True)
    _reassemble_shards(local_dir)
    print(f"[{subset}] -> {local_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO_ID,
                        help=f"Combined HF dataset repo id (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--subset", choices=["corpus", "rq1", "rq2", "all"], default="all",
                        help="Which subset to download (default: all)")
    parser.add_argument("--revision", default=None, help="Optional branch / tag / commit")
    parser.add_argument("--local-dir", type=Path, default=None,
                        help="Override target dir (only valid with a single --subset)")
    args = parser.parse_args()

    subsets = list(DEFAULT_TARGETS) if args.subset == "all" else [args.subset]
    if args.local_dir and len(subsets) != 1:
        raise SystemExit("--local-dir requires a single --subset")

    for subset in subsets:
        target = args.local_dir or DEFAULT_TARGETS[subset]
        download_subset(args.repo, subset, args.revision, target)


if __name__ == "__main__":
    main()
