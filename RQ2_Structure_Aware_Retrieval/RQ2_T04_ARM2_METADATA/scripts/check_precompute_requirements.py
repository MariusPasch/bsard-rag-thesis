"""Programmatic preflight check before running precompute_t04_indices.py.

Verifies that the host has everything the precompute step needs: Python
version, package imports, CUDA, repository paths, AzureDI dump, BSARD DB,
PDF document map, per-doc source PDFs, writable cache root, and that the
T04 corpus loads end-to-end with the expected node-count and linker
coverage for every doc declared in MyDocuments.csv.

The set of docs checked is whatever the active MyDocuments.csv +
pdf_document_map.csv declare — no hard-coded doc_id assumption. The
checker iterates every declared mapping and reports per-doc.

Usage (from the repo root, with the venv active):

    python RQ2_T04_ARM2_METADATA/scripts/check_precompute_requirements.py

Optional overrides (mirror precompute_t04_indices.py):

    python RQ2_T04_ARM2_METADATA/scripts/check_precompute_requirements.py \\
        --azuredi-dir D:/path/to/AzureDI \\
        --pdf-dir D:/path/to/pdfs \\
        --bsard-db D:/path/to/bsard_corpus.db \\
        --pdf-doc-map D:/path/to/pdf_document_map.csv \\
        --cache-root D:/path/to/RQ2_T04_ARM2_METADATA/data

Exit codes:
  0 — all checks passed (or only [WARN]s)
  1 — at least one [FAIL] — do not run the precompute until fixed
"""

from __future__ import annotations

import argparse
import importlib
import os
import sqlite3
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

# Make sibling src/ importable.
for _sibling in _REPO.glob("RQ2_T0*/src"):
    s = str(_sibling)
    if s not in sys.path:
        sys.path.insert(0, s)


# ── Defaults (mirror precompute_t04_indices.py) ──────────────────────────────


DEFAULT_AZUREDI_DIR = _REPO / "RQ2_T04_ARM2_METADATA" / "azuredi"
DEFAULT_PDF_DOC_MAP_CSV = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "csv" / "pdf_document_map.csv"
DEFAULT_BSARD_DB_PATH = _REPO / "dataset_creation_output" / "bsard_corpus.db"
DEFAULT_T04_CACHE_ROOT = _REPO / "RQ2_T04_ARM2_METADATA" / "data"
DEFAULT_PDF_DIR = _REPO / "RQ2_T02_DATA_LOADER" / "data" / "pdfs"

DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"

REQUIRED_AZUREDI_FILES = (
    "MyDocuments.csv",
    "DocumentDefinitions.csv",
    "VectorDB_Documents.json",
    "VectorDBdev_Documents.json",
)
MIN_NODES_PER_DOC = 50  # any kept doc should have at least this many nodes; below = wrong dump


# ── Reporting ────────────────────────────────────────────────────────────────


_FAILS: list[str] = []
_WARNS: list[str] = []


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")
    _WARNS.append(msg)


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    _FAILS.append(msg)


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


# ── Individual checks ────────────────────────────────────────────────────────


def check_python_version() -> None:
    section("1. Python interpreter")
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro} (>= 3.10)")
    else:
        fail(f"Python {v.major}.{v.minor}.{v.micro} is too old; need >= 3.10")
    ok(f"executable: {sys.executable}")


def check_imports() -> None:
    section("2. Required packages")
    requirements = [
        ("torch", None),
        ("sentence_transformers", None),
        ("transformers", None),
        ("faiss", "faiss-cpu (or faiss-gpu)"),
        ("rank_bm25", None),
        ("fitz", "PyMuPDF"),
        ("pandas", None),
        ("numpy", None),
        ("scipy", None),
        ("ollama", None),  # imported by shared.llm at module load
    ]
    for module, install_hint in requirements:
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, "__version__", "?")
            ok(f"{module} {version}")
        except ImportError as exc:
            hint = install_hint or module
            fail(f"{module} not installed — try `pip install {hint}` ({exc})")

    section("2b. Sibling editable installs")
    siblings = [
        ("shared", "RQ2_T01_SHARED"),
        ("data_loader", "RQ2_T02_DATA_LOADER"),
        ("arm1_naive", "RQ2_T03_ARM1_NAIVE"),
        ("arm2_metadata", "RQ2_T04_ARM2_METADATA"),
        ("evaluation", "RQ2_T07_EVALUATION"),
    ]
    for module, sibling in siblings:
        try:
            mod = importlib.import_module(module)
            ok(f"{module} -> {Path(mod.__file__).resolve().parent}")
        except ImportError:
            fail(
                f"{module} not importable — run `pip install -e {sibling}` "
                f"from the repo root with the venv active."
            )


def check_cuda() -> None:
    section("3. CUDA / GPU availability")
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                free, total = torch.cuda.mem_get_info(i)
                ok(f"GPU {i}: {name}  ({free/1e9:.1f}/{total/1e9:.1f} GB free)")
            ok(f"torch CUDA build: {torch.version.cuda}")
        else:
            warn(
                "torch.cuda.is_available() = False — precompute will run on CPU "
                "(~30 hrs for the full plan). Continue if intended."
            )
    except ImportError:
        fail("torch not installed — see [Required packages] above.")


def check_paths(args: argparse.Namespace) -> None:
    section("4. Repository paths and data")

    # AzureDI dump (junction or actual)
    if args.azuredi_dir.exists():
        ok(f"azuredi dir: {args.azuredi_dir}")
        for fname in REQUIRED_AZUREDI_FILES:
            p = args.azuredi_dir / fname
            if p.exists():
                size_kb = p.stat().st_size / 1024
                if size_kb < 1:
                    warn(f"  {fname} present but only {size_kb:.1f} KB — "
                         "a partially-synced/placeholder file? make sure the "
                         "full file is materialised on local disk.")
                else:
                    ok(f"  {fname}  ({size_kb:.0f} KB)")
            else:
                fail(f"  {fname} missing under {args.azuredi_dir}")
    else:
        fail(f"azuredi dir not found: {args.azuredi_dir}")

    # BSARD DB
    if args.bsard_db.exists():
        ok(f"bsard_db: {args.bsard_db}  ({args.bsard_db.stat().st_size/1e6:.1f} MB)")
        try:
            conn = sqlite3.connect(str(args.bsard_db))
            n_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            n_questions = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
            conn.close()
            if n_articles > 1000 and n_questions > 100:
                ok(f"  articles={n_articles}, questions={n_questions}")
            else:
                warn(f"  articles={n_articles}, questions={n_questions} — "
                     "looks under-populated; expected ~40k articles, ~1100 questions.")
        except sqlite3.Error as exc:
            fail(f"  could not query bsard_db: {exc}")
    else:
        fail(f"bsard_db not found: {args.bsard_db}")

    # PDF doc map — list every curated mapping
    if args.pdf_doc_map.exists():
        ok(f"pdf_doc_map: {args.pdf_doc_map}")
        import pandas as pd
        df = pd.read_csv(args.pdf_doc_map, encoding="utf-8")
        if df.empty:
            fail("  pdf_document_map.csv is empty — no docs will resolve to BSARD.")
        else:
            ok(f"  {len(df)} mapping(s):")
            for _, row in df.iterrows():
                ok(f"    {row['pdf_filename']} -> document_id={row['document_id']}")
    else:
        fail(f"pdf_doc_map not found: {args.pdf_doc_map}")

    # PDFs — check each mapped PDF exists on disk under --pdf-dir
    if args.pdf_doc_map.exists() and args.pdf_dir.exists():
        import pandas as pd
        df = pd.read_csv(args.pdf_doc_map, encoding="utf-8")
        for _, row in df.iterrows():
            pdf_path = args.pdf_dir / str(row["pdf_filename"])
            if pdf_path.exists():
                size = pdf_path.stat().st_size
                ok(f"pdf_path: {pdf_path.name}  ({size/1e6:.1f} MB)")
            else:
                warn(
                    f"pdf_path not found: {pdf_path} — precompute for that doc "
                    "will fall back to the AzureDI dump fingerprint. Cache hashes "
                    "computed here will differ from those of any host that has the "
                    "PDF, so a host that does have the PDF may rebuild on first "
                    "retrieval."
                )
    elif not args.pdf_dir.exists():
        warn(f"pdf_dir not found: {args.pdf_dir} — skipping per-PDF presence check.")

    # Cache root (writable?)
    cache_parent = args.cache_root if args.cache_root.exists() else args.cache_root.parent
    if cache_parent.exists() and os.access(cache_parent, os.W_OK):
        ok(f"cache_root writable: {args.cache_root}")
    elif cache_parent.exists():
        fail(f"cache_root not writable: {args.cache_root}")
    else:
        fail(f"cache_root parent does not exist: {args.cache_root.parent}")

    # Disk space — rough headroom estimate
    try:
        import shutil
        free_gb = shutil.disk_usage(cache_parent).free / 1e9
        if free_gb >= 5:
            ok(f"free disk on cache_root volume: {free_gb:.1f} GB (>= 5 GB)")
        else:
            warn(f"only {free_gb:.1f} GB free on cache_root volume — "
                 "FAISS indices for the full plan need ~2-4 GB.")
    except OSError as exc:
        warn(f"could not check disk space: {exc}")


def check_embedding_model_cached(args: argparse.Namespace) -> None:
    section("5. Embedding model availability (offline check)")
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        names = {repo.repo_id for repo in info.repos}
        if args.embedding_model in names:
            size_mb = sum(
                repo.size_on_disk for repo in info.repos
                if repo.repo_id == args.embedding_model
            ) / 1e6
            ok(f"{args.embedding_model} cached locally ({size_mb:.0f} MB)")
        else:
            warn(
                f"{args.embedding_model} NOT in HF cache — first call will "
                "download (~1.1 GB for mE5-large-instruct). Confirm internet "
                "access or pre-download via "
                f"`huggingface-cli download {args.embedding_model}`."
            )
    except ImportError:
        warn("huggingface_hub.scan_cache_dir not available — skipping cache check.")


def check_corpus_load(args: argparse.Namespace) -> None:
    section("6. End-to-end corpus load (smoke)")
    try:
        from arm2_metadata.azuredi_loader import load_azuredi_corpus
        from arm2_metadata.bsard_link import link_corpus_to_bsard
        from arm2_metadata.cache import azuredi_dump_fingerprint
    except ImportError as exc:
        fail(f"arm2_metadata not importable — {exc}")
        return

    try:
        nodes, my_documents, defs = load_azuredi_corpus(
            args.azuredi_dir, args.pdf_doc_map,
        )
    except Exception as exc:
        fail(f"load_azuredi_corpus failed: {exc}")
        return

    if my_documents:
        ok(f"loaded MyDocuments: {sorted(my_documents)} (doc_id=1 hard-dropped in code)")
    else:
        fail("loaded MyDocuments is empty — check that doc_id=1 isn't the only row.")

    # Per-doc node-count check against MIN_NODES_PER_DOC threshold
    from collections import Counter
    per_doc = Counter(n.doc_id for n in nodes)
    for did in sorted(my_documents):
        n_did = per_doc.get(did, 0)
        if n_did == 0:
            fail(f"no AzureDI nodes for doc_id={did} — declared in MyDocuments.csv but absent from VectorDB.")
        elif n_did < MIN_NODES_PER_DOC:
            warn(f"doc_id={did} has only {n_did} nodes (< {MIN_NODES_PER_DOC}) — likely wrong dump.")
        else:
            ok(f"doc_id={did} nodes: {n_did}")

    # Linker pass + coverage per doc
    try:
        nodes = link_corpus_to_bsard(nodes, args.bsard_db)
    except Exception as exc:
        fail(f"link_corpus_to_bsard failed: {exc}")
        return

    for did in sorted(my_documents):
        n_with_bid = sum(1 for n in nodes if n.bsard_ids and n.doc_id == did)
        n_content = sum(1 for n in nodes if n.doc_id == did and not n.is_header
                                           and n.page_content.strip())
        if n_content == 0:
            warn(f"doc_id={did}: no content nodes — nothing to link.")
        elif n_with_bid / n_content >= 0.95:
            ok(f"doc_id={did}: BSARD linker coverage {n_with_bid}/{n_content} content nodes "
               f"({n_with_bid/n_content:.1%}, >= 95%)")
        else:
            warn(f"doc_id={did}: BSARD linker coverage low: {n_with_bid}/{n_content} content nodes "
                 f"({n_with_bid/n_content:.1%}) — expected >= 95%; check linker_version >= 2 is in place.")

    # Fingerprint
    fp = azuredi_dump_fingerprint(args.azuredi_dir)
    ok(f"azuredi_dump_fingerprint: {fp[:24]}…")


# ── Driver ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_precompute_requirements.py",
        description=__doc__.split("\n", 1)[0],
    )
    parser.add_argument("--azuredi-dir", type=Path, default=DEFAULT_AZUREDI_DIR)
    parser.add_argument("--pdf-doc-map", type=Path, default=DEFAULT_PDF_DOC_MAP_CSV)
    parser.add_argument("--bsard-db", type=Path, default=DEFAULT_BSARD_DB_PATH)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR,
                        help="Directory containing per-PDF files named <stem>.pdf. "
                             "The checker derives PDF paths from pdf_document_map.csv.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_T04_CACHE_ROOT)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args(argv)

    print("=" * 72)
    print("  T04 precompute — preflight requirements check")
    print("=" * 72)

    check_python_version()
    check_imports()
    check_cuda()
    check_paths(args)
    check_embedding_model_cached(args)
    check_corpus_load(args)

    section("Summary")
    if _FAILS:
        for f in _FAILS:
            print(f"  [FAIL] {f}")
        print(f"\n{len(_FAILS)} failure(s), {len(_WARNS)} warning(s) — "
              "do NOT run precompute until failures are fixed.")
        return 1
    if _WARNS:
        for w in _WARNS:
            print(f"  [WARN] {w}")
        print(f"\nNo failures, {len(_WARNS)} warning(s) — read each warning, "
              "then proceed with precompute_t04_indices.py if comfortable.")
    else:
        print("  All checks passed. Safe to run precompute_t04_indices.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
