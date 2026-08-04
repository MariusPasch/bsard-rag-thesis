#!/usr/bin/env python
"""Produce the minimal, publication-safe AzureDI bundle shipped in the public release.

The raw AzureDI dump (``VectorDB_Documents.json``, ~316 MB) is **not** safe to
publish as-is: most of its size is per-record embeddings never read by the
code, and it carries several internal identifiers, links, and layout fields
that play no role in reproducing the retrieval results. This script derives
a minimised copy that reproduces Arm 2A / Arm 2C identically while exposing
only what the loader actually consumes.

It is deterministic and never mutates the source.

What it does
------------
* ``VectorDB_Documents.json`` -> sanitised ``VectorDB_Documents.json``:
    - keep only records whose ``payload.metadata.document_id`` is in the kept set
      (kept = MyDocuments.csv ids minus doc_id 1, mirroring ``azuredi_loader.py``)
    - drop fields the code never reads or that are not safe to publish:
        ``_id``, ``vector_full_embeddings``,
        ``payload.metadata.{document_url, image_url, 2D_embeddings,
        2D_embeddings_content_summary, bounding_boxes}``
    - keep exactly the fields ``azuredi_loader.py`` reads:
        top-level ``id``; ``payload.page_content``;
        ``payload.metadata.{id, document_id, parent_id, node_level, node_source,
        document_type, document_status, page_number, is_header, has_requirements,
        path_to_item, content_summary, keywords, level_summary}``
    - ``parent_id`` NaN / extended-JSON -> ``null`` (clean strict JSON)
* ``VectorDBdev_Documents.json`` -> DROPPED (only doc_ids 45/46; never survive
  the filter, so they contribute nothing at runtime)
* ``DocumentDefinitions.csv``    -> DROPPED (only doc_id 2, which is filtered out
  at runtime; ``load_azuredi_corpus`` handles its absence gracefully)
* ``MyDocuments.csv``            -> redacted copy (internal identifier columns blanked)
* ``backups/``                   -> never copied

The result loads through ``arm2_metadata.azuredi_loader.load_azuredi_corpus``
with byte-for-byte identical node counts to the raw dump.

Usage::

    python scripts/sanitise_azuredi.py                       # source/out from defaults
    python scripts/sanitise_azuredi.py --src <AzureDI dir> --out <out dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
DEFAULT_SRC = _REPO / "RQ2_T04_ARM2_METADATA" / "azuredi"   # junction -> AzureDI
DEFAULT_OUT = _REPO / "RQ2_T04_ARM2_METADATA" / "azuredi_sanitised"

# Fields kept inside payload.metadata (exactly what azuredi_loader.py reads).
META_KEEP = [
    "id", "document_id", "parent_id", "node_level", "node_source",
    "document_type", "document_status", "page_number", "is_header",
    "has_requirements", "path_to_item", "content_summary", "keywords",
    "level_summary",
]
# MyDocuments.csv columns to blank (internal identifiers; unused downstream).
MYDOCS_REDACT = {
    "UserId", "DocumentPdfUrl", "CollectionName", "Hash",
    "UploadCost", "UploadTimeMap", "VectorDbIDs",
}


def coerce_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return None if v != v else int(v)          # NaN -> None
    if isinstance(v, int):
        return v
    if isinstance(v, dict):
        for k in ("$numberInt", "$numberLong", "$numberDouble"):
            if k in v:
                try:
                    f = float(v[k])
                    return None if f != f else int(f)
                except (TypeError, ValueError):
                    return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def iter_objects(path: Path):
    """Brace-matched record iterator, mirroring azuredi_loader.iter_objects."""
    raw = path.read_text(encoding="utf-8")
    n = len(raw)
    i = raw.find("{")                              # first record (skip leading '[')
    while i != -1 and i < n:
        start, depth, in_str, esc = i, 0, False, False
        while i < n:
            c = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
            i += 1
        chunk = raw[start:i].replace('{"$numberDouble": "NaN"}', "NaN") \
                            .replace('{ "$numberDouble": "NaN" }', "NaN")
        try:
            yield json.loads(chunk)
        except json.JSONDecodeError:
            try:
                yield json.loads(chunk.replace("NaN", "null"))
            except json.JSONDecodeError:
                pass
        i = raw.find("{", i)


def kept_doc_ids(src: Path) -> set[int]:
    ids: set[int] = set()
    with open(src / "MyDocuments.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d = coerce_int(row.get("DocumentId"))
            if d is not None and d != 1:
                ids.add(d)
    return ids


def sanitise_vectordb(src: Path, out: Path, keep: set[int]) -> dict:
    out_path = out / "VectorDB_Documents.json"
    stats = {"kept": 0, "dropped": 0, "per_doc": {}}
    with open(out_path, "w", encoding="utf-8") as w:
        w.write("[\n")
        first = True
        for rec in iter_objects(src / "VectorDB_Documents.json"):
            payload = rec.get("payload") or {}
            md = payload.get("metadata") or {}
            did = coerce_int(md.get("document_id"))
            if did is None or did not in keep:
                stats["dropped"] += 1
                continue
            new_md = {}
            for k in META_KEEP:
                if k == "parent_id":
                    new_md[k] = coerce_int(md.get("parent_id"))    # NaN -> null
                elif k in md:
                    new_md[k] = md[k]
            new_rec = {
                "id": rec.get("id"),
                "payload": {
                    "page_content": payload.get("page_content", ""),
                    "metadata": new_md,
                },
            }
            w.write(("" if first else ",\n") + json.dumps(new_rec, ensure_ascii=False))
            first = False
            stats["kept"] += 1
            stats["per_doc"][did] = stats["per_doc"].get(did, 0) + 1
        w.write("\n]\n")
    stats["out_bytes"] = out_path.stat().st_size
    return stats


def sanitise_mydocs(src: Path, out: Path) -> None:
    with open(src / "MyDocuments.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    redact_idx = {i for i, h in enumerate(header) if h in MYDOCS_REDACT}
    with open(out / "MyDocuments.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows[1:]:
            if not row:
                continue
            d = coerce_int(row[0]) if row else None
            if d == 1:                                  # drop EU doc if present
                continue
            w.writerow(["" if i in redact_idx else c for i, c in enumerate(row)])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Source AzureDI dir. Default: %(default)s")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output dir. Default: %(default)s")
    args = p.parse_args(argv)

    src, out = args.src, args.out
    if not (src / "VectorDB_Documents.json").exists():
        raise FileNotFoundError(f"No VectorDB_Documents.json under {src}")
    out.mkdir(parents=True, exist_ok=True)

    keep = kept_doc_ids(src)
    print(f"kept doc ids (MyDocuments.csv minus 1): {sorted(keep)}")
    v = sanitise_vectordb(src, out, keep)
    sanitise_mydocs(src, out)
    src_bytes = (src / "VectorDB_Documents.json").stat().st_size
    print(f"VectorDB_Documents.json: kept {v['kept']} recs, dropped {v['dropped']}")
    print(f"  per-doc kept: {dict(sorted(v['per_doc'].items()))}")
    print(f"  size: {src_bytes/1e6:.1f} MB -> {v['out_bytes']/1e6:.2f} MB "
          f"({100*v['out_bytes']/src_bytes:.1f}% of original)")
    print("DROPPED (not written): VectorDBdev_Documents.json, DocumentDefinitions.csv, backups/")
    print(f"MyDocuments.csv: redacted columns {sorted(MYDOCS_REDACT)}")
    print(f"output dir: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
