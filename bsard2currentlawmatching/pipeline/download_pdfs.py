"""Download all 49 Justel consolidated PDFs listed in bsard_full_verify.csv.

PDFs are saved to output/pdfs/. Already-downloaded files are skipped unless
--force is passed.

The input CSV (bsard_full_verify.csv) is ~1 MB and is published alongside the
rest of the dataset on Hugging Face:
    https://huggingface.co/datasets/mpaschalidis/bsard2currentlawmatching
Either pull it via `python scripts/download_from_hf.py` (which puts it at
output/bsard_full_verify.csv) or pass --csv to point at an existing copy.

Usage:
    python pipeline/download_pdfs.py
    python pipeline/download_pdfs.py --csv PATH/TO/bsard_full_verify.csv
    python pipeline/download_pdfs.py --force
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "output" / "bsard_full_verify.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "pdfs"


def url_to_filename(url: str) -> str:
    """Convert a Justel PDF URL to a safe local filename.

    Example:
        https://www.ejustice.just.fgov.be/img_l/pdf/1804/03/21/1804032150_F.pdf
        → img_l_pdf_1804_03_21_1804032150_F.pdf
    """
    # Strip scheme and host, then replace path separators with underscores.
    path = url.split("ejustice.just.fgov.be/")[-1]
    return path.replace("/", "_")


def download_pdfs(
    csv_path: Path,
    out_dir: Path,
    force: bool = False,
    delay: float = 1.0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    urls = df["pdf_url"].dropna().unique().tolist()
    print(f"Found {len(urls)} unique PDF URLs in {csv_path.name}\n")

    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (compatible; BSARD-thesis-dataset-builder/1.0)"
    )

    ok = skipped = failed = 0

    for i, url in enumerate(urls, 1):
        filename = url_to_filename(url)
        dest = out_dir / filename

        if dest.exists() and not force:
            size_kb = dest.stat().st_size // 1024
            print(f"[{i:2d}/{len(urls)}] SKIP  {filename}  ({size_kb} KB)")
            skipped += 1
            continue

        print(f"[{i:2d}/{len(urls)}] {filename} ...", end=" ", flush=True)
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()

            # Verify it looks like a PDF.
            if not resp.content.startswith(b"%PDF"):
                print(f"WARNING — response is not a PDF (got {resp.content[:8]}), skipping")
                failed += 1
                continue

            dest.write_bytes(resp.content)
            size_kb = len(resp.content) // 1024
            print(f"{size_kb} KB")
            ok += 1
            time.sleep(delay)

        except requests.RequestException as exc:
            print(f"ERROR — {exc}")
            failed += 1

    print(f"\nDone. Downloaded: {ok}  Skipped: {skipped}  Failed: {failed}")
    if failed:
        print("Re-run with --force to retry failed downloads.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all 49 Justel consolidated PDFs"
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Path to bsard_full_verify.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Directory to save PDFs (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the file already exists",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between requests (default: 1.0)",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found at {args.csv}")
        return

    download_pdfs(args.csv, args.out_dir, args.force, args.delay)


if __name__ == "__main__":
    main()
