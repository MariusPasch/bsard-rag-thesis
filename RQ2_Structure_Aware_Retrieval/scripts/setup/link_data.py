"""Wire each sub-project's ``data/`` directory to the shared data root.

RQ2's sub-projects expect their large inputs under a local ``data/`` (and, for
T04, ``azuredi/``) directory. These point into the data root populated by
``scripts/download_data.py`` (``$RQ2_DATA_DIR`` or ``<repo>/data``), which is
itself filled from the companion Hugging Face dataset
``Marios-Paschalidis-Thesis/bsard-rag-thesis-data`` (subset ``rq2``).

This script (re)creates those links. On Windows it makes directory junctions
(no admin rights needed); on POSIX it makes symlinks. Pass ``--copy`` to copy
instead of linking (uses more disk but avoids link support issues).

Mapping (repo path -> data-root subdir):
    RQ2_T02_DATA_LOADER/data    -> shared_source
    RQ2_T03_ARM1_NAIVE/data     -> pdf_cache
    RQ2_T07_EVALUATION/data     -> pdf_cache          (shared with T03)
    RQ2_T04_ARM2_METADATA/data  -> RQ2_T04_ARM2_METADATA
    RQ2_T04_ARM2_METADATA/azuredi -> AzureDI
    RQ2_T05_ARM2_PAGEINDEX/data -> RQ2_T05_ARM2_PAGEINDEX

The BSARD corpus DB lives at ``<data root>/bsard_corpus.db``; point code at it
with ``RQ2_BSARD_DB`` (see .env.example) or rely on the resolver defaults.

Usage:
    python scripts/setup/link_data.py
    python scripts/setup/link_data.py --copy
    python scripts/setup/link_data.py --force      # replace existing links
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# repo-relative link path -> data-root-relative target
LINKS: dict[str, str] = {
    "RQ2_T02_DATA_LOADER/data": "shared_source",
    "RQ2_T03_ARM1_NAIVE/data": "pdf_cache",
    "RQ2_T07_EVALUATION/data": "pdf_cache",
    "RQ2_T04_ARM2_METADATA/data": "RQ2_T04_ARM2_METADATA",
    "RQ2_T04_ARM2_METADATA/azuredi": "AzureDI",
    "RQ2_T05_ARM2_PAGEINDEX/data": "RQ2_T05_ARM2_PAGEINDEX",
}


def data_dir() -> Path:
    env = os.environ.get("RQ2_DATA_DIR")
    return Path(env).expanduser() if env else REPO_ROOT / "data"


def _reparse(p: Path) -> bool:
    # Junctions are reparse points, not symlinks; os.path.islink misses them.
    try:
        return bool(os.lstat(p).st_reparse_tag)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


def make_link(link: Path, target: Path, *, copy: bool, force: bool) -> str:
    if link.exists() or link.is_symlink():
        if not force:
            return f"skip (exists)   {link}"
        if link.is_symlink() or _reparse(link):
            link.unlink(missing_ok=True)
        elif link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink(missing_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(target, link)
        return f"copied          {link} <= {target}"
    if os.name == "nt":
        # Directory junction; no admin rights required.
        import subprocess

        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
        )
        return f"junction        {link} -> {target}"
    link.symlink_to(target, target_is_directory=True)
    return f"symlink         {link} -> {target}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy", action="store_true", help="Copy instead of link.")
    parser.add_argument("--force", action="store_true", help="Replace existing links.")
    args = parser.parse_args()

    root = data_dir()
    if not root.exists():
        print(
            f"Data root {root} does not exist. Run scripts/download_data.py first.",
            file=sys.stderr,
        )
        return 1

    rc = 0
    for rel_link, rel_target in LINKS.items():
        link = REPO_ROOT / rel_link
        target = root / rel_target
        if not target.exists():
            print(f"MISSING target  {target}  (for {rel_link})", file=sys.stderr)
            rc = 1
            continue
        try:
            print(make_link(link, target, copy=args.copy, force=args.force))
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED          {link}: {exc}", file=sys.stderr)
            rc = 1

    db = root / "bsard_corpus.db"
    print()
    if db.exists():
        print(f"BSARD DB:  {db}")
        print(f"   set  RQ2_BSARD_DB={db}   (or rely on resolver defaults)")
    else:
        print(f"NOTE: {db} not found — ensure the bundle includes bsard_corpus.db")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
