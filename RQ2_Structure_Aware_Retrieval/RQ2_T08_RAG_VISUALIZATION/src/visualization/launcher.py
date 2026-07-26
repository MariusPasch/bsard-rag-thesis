"""Public launch() helper for spawning the Streamlit viewer from orchestrator or scripts.

Usage (from orchestrator):
    from visualization.launcher import launch
    proc = launch(bundle, mode="retrieval", arm="arm1", retrieval_results=results)
    ...
    proc.terminate()

The bundle and options are serialised to a temporary JSON side-file and passed
to the Streamlit process as a command-line argument:
    streamlit run app.py -- --bundle-json=<tmp>.json
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from data_loader.models import DocumentBundle, RetrievalResult
    from visualization.query_loader import Query


_APP_PATH = Path(__file__).with_name("app.py")


def launch(
    bundle: "DocumentBundle",
    mode: Literal["review", "retrieval"] = "review",
    arm: str = "arm1",
    retrieval_results: "list[RetrievalResult] | None" = None,
    queries: "list[Query] | None" = None,
    arm_index_dir: Path | None = None,
    db_path: Path | None = None,
    config: dict | None = None,
    port: int = 8501,
    open_browser: bool = True,
) -> subprocess.Popen:
    """Spawn `streamlit run app.py` in a subprocess.

    Args:
        bundle:            DocumentBundle to visualise.
        mode:              "review" or "retrieval".
        arm:               Arm identifier ("arm1" | "2A-full" | "2B" | "2C-1hop" | ...).
        retrieval_results: Pre-computed results to display in retrieval mode.
        queries:           Pre-loaded Query objects to populate the query dropdown.
        arm_index_dir:     Root index directory (data/indices/) for fresh retrieval.
        db_path:           Path to bsard_corpus.db (for query loading and GT lookup).
        config:            Optional config dict (unused by viewer; reserved for future use).
        port:              Streamlit server port.
        open_browser:      Whether Streamlit should open the browser automatically.

    Returns:
        Popen handle. Caller is responsible for calling .terminate() when done.
    """
    # Serialise bundle to a temporary JSON file
    bundle_data = _serialise_bundle(bundle, retrieval_results, queries, arm_index_dir, db_path, mode, arm)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="rag_vis_bundle_",
        delete=False,
    )
    json.dump(bundle_data, tmp, default=str, indent=2)
    tmp.close()

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(_APP_PATH),
        "--server.port", str(port),
        "--server.headless", "false" if open_browser else "true",
        # Disable the source watcher: it walks every module's __path__, forcing
        # transformers to import image-processing submodules that need
        # torchvision (absent) and flood the console with tracebacks. Set here
        # too (not just .streamlit/config.toml) so the spawn is CWD-independent.
        "--server.fileWatcherType", "none",
        "--",
        f"--bundle-json={tmp.name}",
        f"--mode={mode}",
        f"--arm={arm}",
    ]

    if arm_index_dir:
        cmd.append(f"--index-dir={arm_index_dir}")
    if db_path:
        cmd.append(f"--db={db_path}")

    return subprocess.Popen(cmd)


def _serialise_bundle(bundle, retrieval_results, queries, arm_index_dir, db_path, mode, arm) -> dict:
    """Convert a DocumentBundle (and optional extras) to a JSON-serialisable dict."""
    try:
        articles_data = [dataclasses.asdict(a) for a in bundle.articles]
    except Exception:
        articles_data = []

    definitions_data = []
    try:
        if bundle.definitions is not None and not bundle.definitions.empty:
            definitions_data = bundle.definitions.to_dict(orient="records")
    except Exception:
        pass

    data: dict = {
        "document_id": bundle.document_id,
        "pdf_path": str(bundle.pdf_path),
        "pdf_filename": bundle.pdf_filename,
        "metadata": bundle.metadata,
        "definitions": definitions_data,
        "articles": articles_data,
        "raw_text": bundle.raw_text,
        "has_azure_extraction": bundle.has_azure_extraction,
        "mode": mode,
        "arm": arm,
    }

    if db_path:
        data["db_path"] = str(db_path)

    if arm_index_dir:
        data["arm_index_dir"] = str(arm_index_dir)

    if retrieval_results:
        data["retrieval_results"] = [dataclasses.asdict(r) for r in retrieval_results]

    if queries:
        data["queries"] = [dataclasses.asdict(q) for q in queries]

    return data
