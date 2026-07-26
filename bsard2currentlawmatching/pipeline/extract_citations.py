"""Phase C — Cross-reference extraction and citation graph construction.

Reads output/linked/articles_linked.jsonl.
For each article:
  1. Scans article_text for Belgian legal citation patterns.
  2. Resolves citations to provisional article_ids where possible.
  3. Adds citation fields to every article record.
  4. Builds the inverse (cited_by) index.
  5. Exports the citation_graph edge list.

Provisional article_ids are assigned 1-based in input order.
Phase D reads this file in the same order and assigns the same sequential
article_ids when inserting into SQLite, so the IDs remain consistent.

Outputs:
  output/linked/articles_with_citations.jsonl  — full article records with
                                                  all citation fields added
  output/linked/citation_graph.jsonl           — resolved directed edge list

Usage:
    python retrieval/extract_citations.py
    python retrieval/extract_citations.py --in-file output/linked/articles_linked.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN_FILE  = PROJECT_ROOT / "output" / "linked" / "articles_linked.jsonl"
DEFAULT_OUT_FILE = PROJECT_ROOT / "output" / "linked" / "articles_with_citations.jsonl"
DEFAULT_GRAPH    = PROJECT_ROOT / "output" / "linked" / "citation_graph.jsonl"

# ── Citation regex patterns ────────────────────────────────────────────────────
# Applied to article_text. All case-insensitive.
# Unicode characters used directly (article text is stored as proper Unicode).

# Pattern 1 — simple article reference: "art. 42", "Art. 1.1.1", "article 3bis"
# NOTE: excludes matches that are structural markers at the start of a line
#       (handled in Phase A, not present in article_text).
_P_SIMPLE = re.compile(
    r"\bart(?:icle)?s?\.?\s+([A-Z0-9][A-Z0-9_./-]*(?:bis|ter|quater|quinquies)?)",
    re.IGNORECASE,
)

# Pattern 2 — range: "articles 12 à 15", "art. 3 et 4", "art. 3 jusqu'au 7"
_P_RANGE = re.compile(
    r"\bart(?:icle)?s?\s+(\d[\w./-]*)\s+(?:et|\xe0|jusqu[\u2019']au?)\s+(\d[\w./-]*)",
    re.IGNORECASE,
)

# Pattern 3 — named law + article: "l'article 3 de la loi du 10 juin 1998"
_P_NAMED = re.compile(
    r"\bl[\u2019']?art(?:icle)?\.?\s+(\d[\w./-]*)\s+de\s+la\s+(?:loi|arr\xeat\xe9|d\xe9cret)",
    re.IGNORECASE,
)

# Pattern 4 — present-code self-reference (relative, do not resolve to ID)
_P_PRESENT = re.compile(
    r"\bdu\s+pr\xe9sent\s+(?:code|titre|chapitre|article)",
    re.IGNORECASE,
)

# Pattern 5 — "l'alinéa précédent", "l'article précédent" (relative, do not resolve)
_P_PREV = re.compile(
    r"\b(?:l[\u2019']al\xedn\xe9a|l[\u2019']article)\s+pr\xe9c\xe9dent\b",
    re.IGNORECASE,
)

# ── Normalisation helpers ──────────────────────────────────────────────────────

def normalise_key(text: str) -> str:
    """Accent-stripped lowercase key for lookup matching."""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


_LEADING_INT_RE = re.compile(r"^(\d+)")
_BASE_ART_RE    = re.compile(r"^([A-Z0-9][A-Z0-9_.]*)-\d", re.IGNORECASE)


def leading_int(art_no: str) -> str | None:
    m = _LEADING_INT_RE.match(str(art_no).strip())
    return m.group(1) if m else None


def base_article_no(art_no: str) -> str | None:
    """Strip sub-paragraph suffix: '1.1.1-1' → '1.1.1', 'I.1-2' → 'I.1'."""
    cleaned = re.sub(r"_[A-Z][A-Z_]+$", "", str(art_no).strip())
    m = _BASE_ART_RE.match(cleaned)
    return m.group(1) if m else None

# ── Citation extraction ────────────────────────────────────────────────────────

def _non_overlapping(matches: list[tuple[int, int, str, str | None, bool]]) \
        -> list[tuple[int, int, str, str | None, bool]]:
    """
    Given a list of (start, end, raw_text, article_no, is_relative), return
    only non-overlapping matches, preferring longer matches when spans overlap.
    """
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))  # sort by start, longest first
    result: list[tuple[int, int, str, str | None, bool]] = []
    last_end = -1
    for m in matches:
        if m[0] >= last_end:
            result.append(m)
            last_end = m[1]
    return result


def extract_raw_citations(text: str) -> list[dict]:
    """
    Extract all citation occurrences from article_text.

    Returns a list of dicts with keys:
      raw_text    — the matched text as it appears in the article
      article_no  — extracted article number (None for relative references)
      is_relative — True for présent/précédent references (no resolution)
    """
    if not text:
        return []

    all_matches: list[tuple[int, int, str, str | None, bool]] = []

    # Pattern 2 (range) is more specific than Pattern 1 — collect first.
    for m in _P_RANGE.finditer(text):
        raw = m.group(0)
        # Add both endpoints of the range as separate citations.
        all_matches.append((m.start(), m.end(), raw, m.group(1), False))
        all_matches.append((m.start(), m.end(), raw, m.group(2), False))

    # Pattern 3 (named law).
    for m in _P_NAMED.finditer(text):
        all_matches.append((m.start(), m.end(), m.group(0), m.group(1), False))

    # Pattern 1 (simple) — collect, will be filtered for overlaps below.
    for m in _P_SIMPLE.finditer(text):
        all_matches.append((m.start(), m.end(), m.group(0), m.group(1), False))

    # Patterns 4 & 5 (relative references).
    for m in _P_PRESENT.finditer(text):
        all_matches.append((m.start(), m.end(), m.group(0), None, True))
    for m in _P_PREV.finditer(text):
        all_matches.append((m.start(), m.end(), m.group(0), None, True))

    # Deduplicate overlapping spans — prefer longer match.
    cleaned = _non_overlapping(all_matches)

    return [
        {"raw_text": raw, "article_no": art_no, "is_relative": is_rel}
        for (_, _, raw, art_no, is_rel) in cleaned
    ]

# ── Resolution ─────────────────────────────────────────────────────────────────

def resolve_article_no(
    art_no: str,
    source_law_code: str | None,
    source_id: int,
    primary_lookup: dict[str, dict[str, int]],
    global_lookup:  dict[str, list[int]],
) -> int | None:
    """
    Try to resolve an article number to a single article_id.

    Resolution chain:
      1. Exact match within the same law_code.
      2. Leading-int match within same law_code (unique only).
      3. Base-article match within same law_code.
      4. Global exact match across all codes (unique only).
      5. Global leading-int match (unique only).

    Self-references (target == source) are excluded.
    """
    norm_no = normalise_key(art_no)
    norm_code = normalise_key(source_law_code) if source_law_code else None

    def _not_self(aid: int) -> bool:
        return aid != source_id

    def _pick(candidates: list[int]) -> int | None:
        valid = [a for a in candidates if _not_self(a)]
        return valid[0] if len(valid) == 1 else None

    # 1. Exact same-law match.
    if norm_code:
        law_dict = primary_lookup.get(norm_code, {})
        aid = law_dict.get(norm_no)
        if aid is not None and _not_self(aid):
            return aid

    # 2. Leading-int same-law match.
    if norm_code:
        li = leading_int(art_no)
        if li:
            law_dict = primary_lookup.get(norm_code, {})
            aid = law_dict.get(li)
            if aid is not None and _not_self(aid):
                return aid

    # 3. Base-article same-law match.
    if norm_code:
        base = base_article_no(art_no)
        if base:
            law_dict = primary_lookup.get(norm_code, {})
            aid = law_dict.get(normalise_key(base))
            if aid is not None and _not_self(aid):
                return aid

    # 4. Global exact match (unique across all codes).
    global_exact = global_lookup.get(norm_no, [])
    resolved = _pick(global_exact)
    if resolved is not None:
        return resolved

    # 5. Global leading-int match.
    li = leading_int(art_no)
    if li:
        global_li = global_lookup.get(li, [])
        resolved = _pick(global_li)
        if resolved is not None:
            return resolved

    return None

# ── Build lookup indices ───────────────────────────────────────────────────────

def build_lookup(
    articles: list[dict],
) -> tuple[dict[str, dict[str, int]], dict[str, list[int]]]:
    """
    Build two indices from the article list:
      primary_lookup[norm_law_code][norm_art_no] → article_id
        — for each (law_code, article_no) key, the canonical article_id is
          the one with article_text_source='bsard_dataset', else the first seen.
      global_lookup[norm_art_no] → [article_id, ...]
        — all article_ids for that article number across all law codes.
    """
    # First pass: group candidates by (norm_code, norm_art_no).
    candidates: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for art in articles:
        art_no = art.get("article_number")
        law_code = art.get("law_code")
        if not art_no or not law_code:
            continue
        key = (normalise_key(law_code), normalise_key(str(art_no)))
        candidates[key].append(art)

    # Second pass: pick canonical article per key.
    primary_lookup: dict[str, dict[str, int]] = defaultdict(dict)
    for (norm_code, norm_no), cands in candidates.items():
        # Prefer bsard_dataset source; otherwise take first.
        canonical = next(
            (c for c in cands if c.get("article_text_source") == "bsard_dataset"),
            cands[0],
        )
        primary_lookup[norm_code][norm_no] = canonical["_provisional_id"]

    # Build global lookup (all article_ids per normalised article_no).
    global_lookup: dict[str, list[int]] = defaultdict(list)
    seen_global: dict[str, set[int]] = defaultdict(set)
    for art in articles:
        art_no = art.get("article_number")
        if not art_no:
            continue
        norm_no = normalise_key(str(art_no))
        aid = art["_provisional_id"]
        if aid not in seen_global[norm_no]:
            global_lookup[norm_no].append(aid)
            seen_global[norm_no].add(aid)

    return dict(primary_lookup), dict(global_lookup)

# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase C: extract and resolve cross-references"
    )
    parser.add_argument("--in-file", type=Path, default=DEFAULT_IN_FILE)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--graph-file", type=Path, default=DEFAULT_GRAPH)
    args = parser.parse_args()

    # ── 1. Load all articles and assign provisional IDs ───────────────────────
    print(f"Loading {args.in_file.name} ...", end=" ", flush=True)
    articles: list[dict] = []
    with args.in_file.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            art = json.loads(line)
            art["_provisional_id"] = i
            articles.append(art)
    print(f"{len(articles):,} articles loaded")

    # ── 2. Build lookup indices ───────────────────────────────────────────────
    print("Building resolution lookup ...", end=" ", flush=True)
    primary_lookup, global_lookup = build_lookup(articles)
    n_primary_keys = sum(len(v) for v in primary_lookup.values())
    print(f"{len(primary_lookup)} law codes, {n_primary_keys:,} keyed articles")

    # ── 3. Extract and resolve citations ──────────────────────────────────────
    print("Extracting and resolving citations ...")

    # citation_graph edges: (source_id, target_id) → citation_text (first seen)
    graph_edges: dict[tuple[int, int], str] = {}
    # Per-article: list of resolved article_ids (deduplicated, preserving order)
    resolved_per_article: list[list[int]] = []

    total_raw = total_resolved = total_relative = 0

    for art in articles:
        source_id  = art["_provisional_id"]
        law_code   = art.get("law_code")
        text       = art.get("article_text") or ""

        raw_cits   = extract_raw_citations(text)
        raw_strings: list[str]  = []
        resolved_ids: list[int] = []
        seen_targets: set[int]  = set()

        for cit in raw_cits:
            raw_strings.append(cit["raw_text"])

            if cit["is_relative"]:
                total_relative += 1
                continue

            if not cit["article_no"]:
                continue

            target_id = resolve_article_no(
                cit["article_no"],
                law_code,
                source_id,
                primary_lookup,
                global_lookup,
            )

            if target_id is not None and target_id not in seen_targets:
                seen_targets.add(target_id)
                resolved_ids.append(target_id)
                edge_key = (source_id, target_id)
                if edge_key not in graph_edges:
                    graph_edges[edge_key] = cit["raw_text"]

        total_raw      += len(raw_strings)
        total_resolved += len(resolved_ids)
        resolved_per_article.append(resolved_ids)

        # Store raw and resolved lists on the article record.
        art["cross_references_raw"]  = raw_strings
        art["cross_reference_ids"]   = resolved_ids
        art["n_outgoing_refs"]       = len(raw_strings)
        art["has_cross_references"]  = 1 if raw_strings else 0

    print(f"  Raw citation occurrences  : {total_raw:,}")
    print(f"  Relative references       : {total_relative:,}")
    print(f"  Resolved citation edges   : {total_resolved:,}")
    print(f"  Unique graph edges        : {len(graph_edges):,}")

    # ── 4. Build inverse index (cited_by) ─────────────────────────────────────
    print("Building cited_by index ...", end=" ", flush=True)
    cited_by: dict[int, list[int]] = defaultdict(list)
    for art, resolved_ids in zip(articles, resolved_per_article):
        source_id = art["_provisional_id"]
        for target_id in resolved_ids:
            cited_by[target_id].append(source_id)

    # Attach cited_by to every article.
    for art in articles:
        aid = art["_provisional_id"]
        cb  = sorted(cited_by.get(aid, []))
        art["cited_by_ids"] = cb
        art["n_cited_by"]   = len(cb)
    print("done")

    # ── 5. Write articles_with_citations.jsonl ────────────────────────────────
    print(f"Writing {args.out_file.name} ...", end=" ", flush=True)
    tmp = args.out_file.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for art in articles:
            rec = {k: v for k, v in art.items() if not k.startswith("_")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(args.out_file)
    print(f"{args.out_file.stat().st_size // 1024:,} KB")

    # ── 6. Write citation_graph.jsonl ──────────────────────────────────────────
    print(f"Writing {args.graph_file.name} ...", end=" ", flush=True)
    tmp = args.graph_file.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for edge_id, ((src, tgt), cit_text) in enumerate(graph_edges.items(), start=1):
            edge = {
                "edge_id":       edge_id,
                "source_id":     src,
                "target_id":     tgt,
                "citation_text": cit_text,
                "resolved":      1,
            }
            f.write(json.dumps(edge, ensure_ascii=False) + "\n")
    tmp.replace(args.graph_file)
    print(f"{len(graph_edges):,} edges, {args.graph_file.stat().st_size // 1024:,} KB")

    # ── 7. Summary stats ──────────────────────────────────────────────────────
    arts_with_refs = sum(1 for a in articles if a["n_outgoing_refs"] > 0)
    arts_with_cited = sum(1 for a in articles if a["n_cited_by"] > 0)
    max_out = max(a["n_outgoing_refs"] for a in articles)
    max_in  = max(a["n_cited_by"] for a in articles)
    print(f"\nCitation graph summary:")
    print(f"  Articles with outgoing refs : {arts_with_refs:,}")
    print(f"  Articles cited by others    : {arts_with_cited:,}")
    print(f"  Max outgoing refs           : {max_out}")
    print(f"  Max in-degree               : {max_in}")
    print(f"\nPhase C complete.")


if __name__ == "__main__":
    main()
