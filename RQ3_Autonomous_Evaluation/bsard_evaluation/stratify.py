from __future__ import annotations

def filter_by(
    qrels: dict,
    strata: dict,
    field: str,
    value: str,
) -> dict:
    """Return qrels subset where strata[qid][field] == value."""
    return {
        qid: rel
        for qid, rel in qrels.items()
        if strata.get(qid, {}).get(field) == value
    }
