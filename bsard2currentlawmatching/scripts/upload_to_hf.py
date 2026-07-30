"""Upload the local ./output/ artifacts to the bsard2currentlawmatching HF dataset.

Mirrors the local `output/` directory to a Hugging Face dataset repo so the
SQLite databases, Parquet exports, JSONL exports, and source PDFs can be
downloaded by anyone via `scripts/download_from_hf.py`.

Authentication: run `hf auth login` once, or set the HF_TOKEN env var
to a token that has write access to the target repo.

Usage:
    python scripts/upload_to_hf.py                       # dry-run plan first
    python scripts/upload_to_hf.py --confirm             # actually upload
    python scripts/upload_to_hf.py --confirm --create    # also create the repo
    python scripts/upload_to_hf.py --repo OTHER/REPO --confirm

Requires `huggingface_hub` (already pinned in requirements.txt).
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import time
from pathlib import Path

import requests
import requests.adapters
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import HfHubHTTPError

# Force a wall-clock timeout on every HTTP call huggingface_hub makes.
# Without this, a stalled TCP socket (Windows-flaky on large LFS uploads)
# blocks the python process forever — the upload progress bar freezes at a
# fixed byte count with the process at 0% CPU, and our retry loop never fires
# because no exception is raised.  With this patch a read-stall of >120s
# raises requests.exceptions.ReadTimeout, which the retry loop catches.
_CONNECT_TIMEOUT_S = 30
_READ_TIMEOUT_S = 120
_orig_adapter_send = requests.adapters.HTTPAdapter.send


def _patched_send(self, request, *, stream=False, timeout=None, verify=True, cert=None, proxies=None):
    if timeout is None:
        timeout = (_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S)
    return _orig_adapter_send(self, request, stream=stream, timeout=timeout,
                              verify=verify, cert=cert, proxies=proxies)


requests.adapters.HTTPAdapter.send = _patched_send

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default companion dataset on the Hugging Face Hub. Override with BSARD_HF_REPO.
DEFAULT_REPO_ID = "mpaschalidis/bsard2currentlawmatching"
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "output"

# Always ignored on upload:
#   - local SQLite WAL/SHM sidecars and OS junk
#   - cache/ and logs/ (local-only working directories)
#   - artifacts that belong to the downstream retrieval project, not this corpus project:
#     embeddings/, results/, llm_judge_cache_*.json
DEFAULT_IGNORE = [
    "*.db-wal",
    "*.db-shm",
    ".DS_Store",
    "Thumbs.db",
    "cache/**",
    "logs/**",
    "embeddings/**",
    "results/**",
    "llm_judge_cache_*.json",
]


def human_bytes(n: int) -> str:
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
        if pat.endswith("/**") and (
            rel_posix.startswith(pat[:-3] + "/") or rel_posix == pat[:-3]
        ):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("BSARD_HF_REPO", DEFAULT_REPO_ID),
                        help=f"HF dataset repo id (default: $BSARD_HF_REPO or {DEFAULT_REPO_ID})")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR,
                        help=f"Local source directory (default: {DEFAULT_LOCAL_DIR})")
    parser.add_argument("--commit-message", default="Sync local output/ to HF dataset",
                        help="Commit message to record on the dataset repo")
    parser.add_argument("--create", action="store_true",
                        help="Create the dataset repo if it does not exist")
    parser.add_argument("--private", action="store_true",
                        help="Make the repo private when creating (default: public)")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually upload. Without this flag, only prints a plan.")
    args = parser.parse_args()

    if not args.local_dir.exists():
        raise SystemExit(f"Local dir not found: {args.local_dir}")

    all_files = [p for p in args.local_dir.rglob("*") if p.is_file()]
    kept, skipped = [], []
    for p in all_files:
        rel = p.relative_to(args.local_dir).as_posix()
        (skipped if is_ignored(rel, DEFAULT_IGNORE) else kept).append(p)
    total_keep = sum(p.stat().st_size for p in kept)
    total_skip = sum(p.stat().st_size for p in skipped)
    print(f"Local source : {args.local_dir}")
    print(f"Target repo  : {args.repo} (dataset)")
    print(f"Ignored      : {', '.join(DEFAULT_IGNORE)}")
    print(f"Will upload  : {len(kept)} files ({human_bytes(total_keep)})")
    print(f"Will skip    : {len(skipped)} files ({human_bytes(total_skip)})")

    if not args.confirm:
        print("\nDry run only. Re-run with --confirm to upload.")
        return

    api = HfApi()
    if args.create:
        create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
        print(f"Ensured dataset repo exists: {args.repo}")

    # Sequential per-entry upload. Each top-level entry in `output/` (single
    # file or whole subdirectory) becomes its own bounded transaction with
    # retries. A network stall in one entry cannot lock up the rest; rerunning
    # the script skips files HfApi already has on the server.
    chunks = sorted(args.local_dir.iterdir(),
                    key=lambda p: (sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                                   if p.is_dir() else p.stat().st_size))
    max_retries = 5
    failed: list[str] = []

    for i, entry in enumerate(chunks, 1):
        rel = entry.relative_to(args.local_dir).as_posix()
        if entry.is_file():
            if is_ignored(rel, DEFAULT_IGNORE):
                print(f"[{i:2d}/{len(chunks)}] SKIP   {rel}")
                continue
            sz = human_bytes(entry.stat().st_size)
            label = f"file   {rel} ({sz})"
        else:
            files = [p for p in entry.rglob('*') if p.is_file()
                     and not is_ignored(p.relative_to(args.local_dir).as_posix(), DEFAULT_IGNORE)]
            if not files:
                print(f"[{i:2d}/{len(chunks)}] SKIP   {rel}/ (all ignored)")
                continue
            total = sum(p.stat().st_size for p in files)
            label = f"folder {rel}/ ({len(files)} files, {human_bytes(total)})"

        for attempt in range(1, max_retries + 1):
            print(f"[{i:2d}/{len(chunks)}] UPLOAD {label}  [attempt {attempt}/{max_retries}]", flush=True)
            try:
                if entry.is_file():
                    api.upload_file(
                        path_or_fileobj=str(entry),
                        path_in_repo=rel,
                        repo_id=args.repo,
                        repo_type="dataset",
                        commit_message=f"Add {rel}",
                    )
                else:
                    api.upload_folder(
                        repo_id=args.repo,
                        repo_type="dataset",
                        folder_path=str(entry),
                        path_in_repo=rel,
                        commit_message=f"Add {rel}/",
                        ignore_patterns=DEFAULT_IGNORE,
                    )
                print(f"[{i:2d}/{len(chunks)}] OK     {rel}", flush=True)
                break
            except (HfHubHTTPError, OSError, ConnectionError, TimeoutError,
                    requests.exceptions.RequestException) as exc:
                wait = min(2 ** attempt, 60)
                print(f"[{i:2d}/{len(chunks)}] FAIL   {rel}: {type(exc).__name__}: {exc}", flush=True)
                if attempt == max_retries:
                    failed.append(rel)
                    print(f"[{i:2d}/{len(chunks)}] GIVEUP {rel} after {max_retries} attempts", flush=True)
                else:
                    print(f"[{i:2d}/{len(chunks)}] WAIT   {wait}s before retry", flush=True)
                    time.sleep(wait)

    print("\n" + "=" * 60)
    if failed:
        print(f"Upload finished with {len(failed)} failed entries:")
        for rel in failed:
            print(f"  - {rel}")
        print("Re-run the script to retry the failed entries.")
        raise SystemExit(1)
    print("Upload complete. All entries succeeded.")


if __name__ == "__main__":
    main()
