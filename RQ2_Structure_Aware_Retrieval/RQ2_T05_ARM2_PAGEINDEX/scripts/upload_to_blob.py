"""Upload a local file to Azure Blob via a container-level SAS URL.

Designed for the T05 round-trip:
  * upload an Azure-side bundle (built by ``prepare_azure_bundle.py``);
  * later, download per-query results back via ``--mode download``.

The SAS URL must be a **container-level** SAS with at minimum:
  read, write, list, create  (write+create are needed for upload).

Pass it via ``--sas-url`` or set the env var
``AZURE_CONTAINER_SAS_URL`` so the secret never lands in shell history.

Usage (upload):
    set AZURE_CONTAINER_SAS_URL=https://<account>.blob.core.windows.net/<container>?<sig>
    python scripts/upload_to_blob.py upload \\
        --file data/azure_bundles/t04_doc2.zip \\
        --blob-name t05_azure_bundles/t04_doc2.zip

Usage (download a prefix back):
    python scripts/upload_to_blob.py download \\
        --prefix t05_results/t04_doc2 \\
        --out data/results/t04_doc2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _container(sas_url: str | None):
    sas_url = sas_url or os.environ.get("AZURE_CONTAINER_SAS_URL", "")
    if not sas_url:
        raise SystemExit(
            "No SAS URL. Pass --sas-url or set AZURE_CONTAINER_SAS_URL."
        )
    from azure.storage.blob import ContainerClient
    return ContainerClient.from_container_url(sas_url)


def cmd_upload(args: argparse.Namespace) -> int:
    log = logging.getLogger("upload")
    src = Path(args.file)
    if not src.exists():
        log.error("File not found: %s", src)
        return 2

    container = _container(args.sas_url)
    blob_name = args.blob_name or f"{src.parent.name}/{src.name}"
    log.info("Uploading %s -> %s (%.1f KB)", src, blob_name, src.stat().st_size / 1024)
    with src.open("rb") as f:
        container.get_blob_client(blob_name).upload_blob(f, overwrite=True)

    # Verify by listing the prefix.
    listed = list(
        container.list_blobs(name_starts_with=blob_name)
    )
    matched = [b for b in listed if b.name == blob_name]
    if matched:
        b = matched[0]
        log.info("Verified: %s  (%d bytes)", b.name, b.size)
        print(f"\nblob_name: {blob_name}\nsize:      {b.size} bytes")
    else:
        log.warning(
            "Upload returned but blob not found in list_blobs - check the SAS "
            "has list permission."
        )
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    log = logging.getLogger("download")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    container = _container(args.sas_url)
    prefix = args.prefix.rstrip("/") + "/"
    n = 0
    for blob in container.list_blobs(name_starts_with=prefix):
        rel = blob.name[len(prefix):]
        target = out_dir / rel
        if target.exists() and target.stat().st_size == blob.size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            f.write(container.get_blob_client(blob).download_blob().readall())
        n += 1
    log.info("Downloaded %d new files to %s", n, out_dir)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    container = _container(args.sas_url)
    prefix = args.prefix.rstrip("/") + "/" if args.prefix else None
    print(f"Container blobs{(' under prefix ' + prefix) if prefix else ''}:")
    total = 0
    for b in container.list_blobs(name_starts_with=prefix):
        total += 1
        print(f"  {b.name:<70}  {b.size:>10} B")
    print(f"\n{total} blobs")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="upload_to_blob")
    parser.add_argument(
        "--sas-url", default=None,
        help="Container-level SAS URL. Falls back to env AZURE_CONTAINER_SAS_URL.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("upload")
    p_up.add_argument("--file", required=True, type=Path)
    p_up.add_argument("--blob-name", default=None,
                      help="Target path inside the container "
                           "(default: <parent_dir>/<filename>).")
    p_up.set_defaults(func=cmd_upload)

    p_dn = sub.add_parser("download")
    p_dn.add_argument("--prefix", required=True,
                      help="Blob name prefix (e.g. t05_results/t04_doc2).")
    p_dn.add_argument("--out", required=True, type=Path,
                      help="Local target directory.")
    p_dn.set_defaults(func=cmd_download)

    p_ls = sub.add_parser("list")
    p_ls.add_argument("--prefix", default=None)
    p_ls.set_defaults(func=cmd_list)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
