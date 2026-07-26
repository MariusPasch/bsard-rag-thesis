"""
Download T4.0-hybrid results from Azure Blob Storage after the Azure notebook run.

Usage:
    python scripts/setup/download_tier40_hybrid_results.py

Requires: pip install azure-storage-blob
"""
from __future__ import annotations
from pathlib import Path
import os

AZURE_CONTAINER_SAS_URL = os.environ.get('AZURE_CONTAINER_SAS_URL', '')
if not AZURE_CONTAINER_SAS_URL:
    raise SystemExit(
        'Set the AZURE_CONTAINER_SAS_URL environment variable to a Blob SAS URL '
        'before running this script.'
    )

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR   = PROJECT_ROOT / 'output'

DOWNLOADS = [
    (
        'results/agentic/llm_judge/llm_rerank/llm_rerank_binary_top50_hybrid_rrf_k60_test.json',
        'results/agentic/llm_judge/llm_rerank/llm_rerank_binary_top50_hybrid_rrf_k60_test.json',
        True,
    ),
    (
        'llm_judge_cache_binary_test_tok1000.json',
        'llm_judge_cache_binary_test_tok1000.json',
        False,
    ),
]


def _download(client, blob_name: str, dest: Path, required: bool) -> bool:
    try:
        bc         = client.get_blob_client(blob_name)
        total_size = bc.get_blob_properties()['size']
    except Exception:
        tag = '[REQUIRED]' if required else '[optional]'
        print(f'  {tag} {blob_name} — not found in blob')
        if required:
            raise FileNotFoundError(f'Required blob missing: {blob_name}')
        return False

    existing = dest.stat().st_size if dest.exists() else 0
    if existing == total_size:
        print(f'  Already complete: {dest.name} ({total_size / 1e6:.1f} MB)')
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f'  Downloading {dest.name} ({total_size / 1e6:.2f} MB) ...', end='', flush=True)
    with open(dest, 'wb') as f:
        for chunk in bc.download_blob().chunks():
            f.write(chunk)
    print(' done')
    return True


def main() -> None:
    from azure.storage.blob import ContainerClient
    client = ContainerClient.from_container_url(AZURE_CONTAINER_SAS_URL)
    for blob_name, local_rel, required in DOWNLOADS:
        _download(client, blob_name, OUTPUT_DIR / local_rel, required)
    print('\nAll downloads complete.')


if __name__ == '__main__':
    main()
