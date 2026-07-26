"""Upload the four components' local data into ONE combined Hugging Face dataset.

Target layout on the Hub (repo: mpaschalidis/bsard-rag-thesis-data, dataset):

    corpus/   ← bsard2currentlawmatching data root  (the SQLite corpus + exports)
    rq1/      ← RQ1_Retrieval_Methods data root              (corpus DB, embeddings, results)
    rq2/      ← RQ2_Structure_Aware_Retrieval data root           (per-arm data, results)
    # rq3 reuses the rq1 bundle — nothing separate to upload.

Each subset is uploaded from a LOCAL data directory (the same place each
component reads/writes its artefacts). Defaults are resolved from the per-
component env vars, falling back to the in-repo default data root:

    corpus : $CORPUS_DATA_DIR  | <repo>/bsard2currentlawmatching/output
    rq1    : $BSARD_DATA_DIR    | <repo>/RQ1_Retrieval_Methods/output
    rq2    : $RQ2_DATA_DIR      | <repo>/RQ2_Structure_Aware_Retrieval/data

Authentication: run `huggingface-cli login` once, or set HF_TOKEN to a token
with write access to the target repo.

Usage:
    python data_tooling/upload_combined_hf.py                      # dry-run plan (all subsets)
    python data_tooling/upload_combined_hf.py --create --confirm   # create repo + upload all
    python data_tooling/upload_combined_hf.py --only corpus --confirm
    python data_tooling/upload_combined_hf.py --only rq2 --rq2-dir D:/data/rq2 --confirm

The per-entry retry/timeout loop is hardened for flaky large-file (LFS) uploads
on Windows. Re-running skips files the Hub already has.

Requires `huggingface_hub`.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import threading
import time
from pathlib import Path

import requests
import requests.adapters
from huggingface_hub import CommitOperationAdd, HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

# ── Force a wall-clock timeout on every HTTP call huggingface_hub makes. ───────
# Without this, a stalled TCP socket (Windows-flaky on large LFS uploads) blocks
# the process forever; the retry loop never fires because no exception is raised.
_CONNECT_TIMEOUT_S = 30
_READ_TIMEOUT_S = 120
_orig_adapter_send = requests.adapters.HTTPAdapter.send


def _patched_send(self, request, *, stream=False, timeout=None, verify=True, cert=None, proxies=None):
    if timeout is None:
        timeout = (_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S)
    return _orig_adapter_send(self, request, stream=stream, timeout=timeout,
                              verify=verify, cert=cert, proxies=proxies)


requests.adapters.HTTPAdapter.send = _patched_send

MONO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_ID = os.environ.get("BSARD_HF_COMBINED_REPO", "mpaschalidis/bsard-rag-thesis-data")

# Patterns ignored for every subset (sidecars + OS/editor junk).
# ".cache/**" matters: HF forbids committing any path under a ".cache/" folder,
# and snapshot_download / hf tooling leaves a local ".cache/huggingface/" behind.
_COMMON_IGNORE = [
    "*.db-wal", "*.db-shm", ".DS_Store", "Thumbs.db",
    "__pycache__/**", "*.pyc", ".git/**", ".ipynb_checkpoints/**",
    ".cache/**",
]


def _subset_config(args) -> dict[str, dict]:
    """subset -> {local_dir, ignore}. local_dir from CLI > env > in-repo default."""
    return {
        "corpus": {
            "local_dir": Path(args.corpus_dir or os.environ.get("CORPUS_DATA_DIR")
                              or MONO_ROOT / "bsard2currentlawmatching" / "output"),
            # Publish only the authoritative corpus artefacts (the 2 DBs, 3 small
            # parquets, verification CSV, stats, id-maps, question_analysis/).
            # Everything else under output/ is regenerable or belongs to an RQ.
            "ignore": _COMMON_IGNORE + [
                ".cache/**", "cache/**", "logs/**", "embeddings/**", "results/**",
                "linked/**", "pdfs/**", "extracted/**",
                "bsard_articles.parquet", "bsard_articles.jsonl",
                "bsard_articles_only.parquet", "bsard_articles_only.jsonl",
                "bsard_articles_clean.jsonl",
                "llm_judge_cache_*.json",
            ],
        },
        "rq1": {
            "local_dir": Path(args.rq1_dir or os.environ.get("BSARD_DATA_DIR")
                              or MONO_ROOT / "RQ1_Retrieval_Methods" / "output"),
            "ignore": _COMMON_IGNORE + ["logs/**"],
        },
        "rq2": {
            "local_dir": Path(args.rq2_dir or os.environ.get("RQ2_DATA_DIR")
                              or MONO_ROOT / "RQ2_Structure_Aware_Retrieval" / "data"),
            # Exclude local-only dev/backup artefacts not part of the published
            # bundle. (Other local-only stragglers can be excluded per-run with
            # --extra-ignore; do NOT broadly glob names like "retrieval_pool"
            # that also occur in published paths.)
            "ignore": _COMMON_IGNORE + [
                "logs/**", ".cache/**", "AzureDI/backups/**",
                "AzureDI/DocumentDefinitions.csv", "AzureDI/VectorDBdev_Documents.json",
            ],
        },
    }


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def is_ignored(rel_posix: str, patterns: list[str]) -> bool:
    """Mirror HfApi.upload_folder semantics for ignore_patterns matching."""
    name = rel_posix.split("/")[-1]
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat) or fnmatch.fnmatch(name, pat):
            return True
        if pat.endswith("/**") and (rel_posix.startswith(pat[:-3] + "/") or rel_posix == pat[:-3]):
            return True
    return False


# Per-call wall-clock budget. Some connections HANG mid-transfer without ever
# raising (no bytes move, no exception). A daemon-thread watchdog turns that into
# a TimeoutError so the retry loop fires. Generous enough for an ~8 MB part/batch.
CALL_TIMEOUT_S = 150


def _call_with_timeout(fn, timeout: float):
    """Run fn() in a daemon thread; raise TimeoutError if it hangs past timeout.
    A hung thread leaks but, being a daemon, never blocks process exit."""
    box: dict = {}

    def target():
        try:
            box["ok"] = fn()
        except BaseException as e:  # noqa: BLE001 - propagated to caller
            box["exc"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"call exceeded {timeout:.0f}s (stalled transfer)")
    if "exc" in box:
        raise box["exc"]
    return box.get("ok")


def _retry(label: str, fn, max_retries: int = 8) -> bool:
    """Run fn() with a hang-watchdog and exponential-backoff retries.

    Broad except on purpose: huggingface_hub raises httpx.* errors (e.g.
    ReadError on a dropped connection), and stalled transfers surface as
    TimeoutError from the watchdog. All calls here are resumable/idempotent.
    """
    for attempt in range(1, max_retries + 1):
        try:
            _call_with_timeout(fn, CALL_TIMEOUT_S)
            return True
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            wait = min(2 ** attempt, 30)
            print(f"  FAIL {label}: {type(exc).__name__}: {exc}", flush=True)
            if attempt == max_retries:
                print(f"  GIVEUP {label} after {max_retries} attempts", flush=True)
                return False
            print(f"  WAIT {wait}s before retry", flush=True)
            time.sleep(wait)
    return False


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def upload_subset(api: HfApi, repo: str, subset: str, local_dir: Path,
                  ignore: list[str], confirm: bool, shard_over_bytes: int = 0) -> list[str]:
    """Upload local_dir into <repo>:<subset>/, resumable.

    Files larger than ``shard_over_bytes`` (when >0) are split into
    ``<rel>.partNNN`` byte-shards of that size, recorded in ``sharded_files.json``,
    and reassembled by data_tooling/download_combined_hf.py. This keeps every uploaded
    object small — essential on connections that drop large transfers. Small files
    go up in one resumable upload_folder commit (ignores are relative to local_dir).
    """
    if not local_dir.exists():
        print(f"[{subset}] SKIP - local dir not found: {local_dir}")
        return []

    kept = [p for p in local_dir.rglob("*") if p.is_file()
            and not is_ignored(p.relative_to(local_dir).as_posix(), ignore)]
    large = [p for p in kept if shard_over_bytes and p.stat().st_size > shard_over_bytes]
    small = [p for p in kept if p not in set(large)]
    total = sum(p.stat().st_size for p in kept)
    print(f"[{subset}] {local_dir}")
    print(f"[{subset}]   -> {repo}:{subset}/  ({len(kept)} files, {human_bytes(total)}; "
          f"{len(large)} sharded > {shard_over_bytes // (1<<20)}MB)")
    if not confirm:
        return []

    failed: list[str] = []
    try:
        existing = set(api.list_repo_files(repo, repo_type="dataset"))
    except Exception:
        existing = set()

    # ---- small files: size-bounded BATCHED commits ----
    # One giant upload_folder commit never lands on a flaky link (it only commits
    # at the very end). Batching into <=BATCH_BYTES / BATCH_FILES commits means
    # each batch lands independently and a drop only retries the current batch.
    small_targets = [(p, f"{subset}/{p.relative_to(local_dir).as_posix()}") for p in small]
    todo = [(p, r) for p, r in small_targets if r not in existing]
    BATCH_BYTES = 8 * (1 << 20)
    BATCH_FILES = 25
    batches: list[list] = []
    cur: list = []
    cur_sz = 0
    for p, r in todo:
        sz = p.stat().st_size
        if cur and (cur_sz + sz > BATCH_BYTES or len(cur) >= BATCH_FILES):
            batches.append(cur); cur, cur_sz = [], 0
        cur.append((p, r)); cur_sz += sz
    if cur:
        batches.append(cur)
    print(f"[{subset}] small files: {len(small)} ({len(todo)} to upload) "
          f"in {len(batches)} batches", flush=True)
    for bi, b in enumerate(batches, 1):
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=str(p)) for p, r in b]
        sz = sum(p.stat().st_size for p, _ in b)
        label = f"{subset}/ small batch {bi}/{len(batches)} ({len(b)} files, {human_bytes(sz)})"
        print(f"[{subset}] {label}", flush=True)
        if not _retry(label, lambda ops=ops, bi=bi: api.create_commit(
                repo_id=repo, repo_type="dataset", operations=ops,
                commit_message=f"Add {subset}/ small files (batch {bi})")):
            failed.append(label)

    # ---- large files: byte-shards uploaded as <=shard_over_bytes parts ----
    manifest: dict[str, dict] = {}
    PART = shard_over_bytes
    for p in sorted(large, key=lambda x: x.stat().st_size, reverse=True):
        rel = p.relative_to(local_dir).as_posix()
        size = p.stat().st_size
        nparts = (size + PART - 1) // PART
        manifest[rel] = {"parts": nparts, "size": size, "sha256": _sha256(p)}
        print(f"[{subset}] shard {rel} -> {nparts} parts ({human_bytes(size)})", flush=True)
        with open(p, "rb") as f:
            for i in range(nparts):
                part_repo = f"{subset}/{rel}.part{i:03d}"
                if part_repo in existing:   # already uploaded — don't re-read bytes
                    continue
                f.seek(i * PART)
                data = f.read(PART)
                ok = _retry(f"{rel}.part{i:03d}", lambda pr=part_repo, d=data: api.upload_file(
                    path_or_fileobj=d, path_in_repo=pr, repo_id=repo, repo_type="dataset",
                    commit_message=f"Add {pr}"))
                if not ok:
                    failed.append(part_repo)
        print(f"[{subset}] OK {rel}", flush=True)

    # ---- manifest (lists only the sharded files) ----
    if manifest:
        man = json.dumps(manifest, indent=2).encode()
        if not _retry(f"{subset}/sharded_files.json", lambda: api.upload_file(
                path_or_fileobj=man, path_in_repo=f"{subset}/sharded_files.json",
                repo_id=repo, repo_type="dataset",
                commit_message=f"Add {subset}/sharded_files.json")):
            failed.append(f"{subset}/sharded_files.json")

    print(f"[{subset}] {'OK' if not failed else 'INCOMPLETE'}", flush=True)
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=DEFAULT_REPO_ID,
                        help=f"Combined HF dataset repo id (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--only", choices=["corpus", "rq1", "rq2"], action="append",
                        help="Upload only this subset (repeatable). Default: all.")
    parser.add_argument("--corpus-dir", help="Local data dir for the corpus subset")
    parser.add_argument("--rq1-dir", help="Local data dir for the rq1 subset")
    parser.add_argument("--rq2-dir", help="Local data dir for the rq2 subset")
    parser.add_argument("--extra-ignore", nargs="*", default=[],
                        help="Extra ignore patterns / exact rel-paths appended to every "
                             "selected subset (e.g. local-only stragglers).")
    parser.add_argument("--create", action="store_true", help="Create the dataset repo if missing")
    parser.add_argument("--private", action="store_true", help="Make the repo private when creating")
    parser.add_argument("--shard-over-mb", type=int, default=0,
                        help="Split files larger than this many MB into <=N MB byte-shards "
                             "(reassembled by download_combined_hf.py). 0 = never shard. "
                             "Use on connections that drop large transfers, e.g. 20.")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually upload. Without this flag, only prints a plan.")
    args = parser.parse_args()

    cfg = _subset_config(args)
    subsets = args.only or ["corpus", "rq1", "rq2"]

    print(f"Combined repo : {args.repo} (dataset)")
    print(f"Subsets       : {', '.join(subsets)}")
    print("-" * 60)

    if args.confirm and args.create:
        create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"Ensured dataset repo exists: {args.repo}")

    api = HfApi()
    shard_bytes = args.shard_over_mb * (1 << 20)
    all_failed: list[str] = []
    for subset in subsets:
        c = cfg[subset]
        ignore = c["ignore"] + list(args.extra_ignore)
        all_failed += upload_subset(api, args.repo, subset, c["local_dir"], ignore,
                                    args.confirm, shard_over_bytes=shard_bytes)
        print("-" * 60)

    if not args.confirm:
        print("\nDry run only. Re-run with --confirm to upload (add --create the first time).")
        return
    if all_failed:
        print(f"\nFinished with {len(all_failed)} failed entries:")
        for rel in all_failed:
            print(f"  - {rel}")
        print("Re-run to retry the failed entries.")
        raise SystemExit(1)
    print("\nUpload complete. All entries succeeded.")


if __name__ == "__main__":
    main()
