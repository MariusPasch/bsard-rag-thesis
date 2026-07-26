"""Export canonical Arm2C (agentic ReAct tree-navigation) runs into a per-doc
layout the T08 viewer can read uniformly.

Arm2C (an agentic ReAct tree-navigation prototype) is out of scope for the
main three-arm comparison; this exporter exists only to surface its runs in the
T08 viewer. Arm2C results live under ``experiments/arm2c/`` (git side), outside
the per-doc ``data/`` layout the viewer's loaders expect. This script
normalises the canonical ``enriched_rerank`` runs for the 5 curated PDFs into
RetrievalResult-shaped per-query JSONs plus a copy of the deep tree, written to
T04's ``data/`` link into the shared data root:

    RQ2_T04_ARM2_METADATA/data/_arm2c/<stem>/results/q<qid>.json
    RQ2_T04_ARM2_METADATA/data/_arm2c/<stem>/deep_tree.json

T08 then reads this via ``t04c_loader.py`` exactly like it reads T04/T05.

Idempotent — overwrites on every run. Read-only w.r.t. the experiment runs.

Run:
    python experiments/arm2c/export_for_t08.py
"""

from __future__ import annotations

import json
from pathlib import Path

# experiments/arm2c/export_for_t08.py -> RQ2_T04_ARM2_METADATA/
_HERE = Path(__file__).resolve().parent
_ARM2C_ROOT = _HERE                          # .../experiments/arm2c
_T04_ROOT = _HERE.parent.parent              # .../RQ2_T04_ARM2_METADATA

# Canonical mode only; the 5 curated PDF stems (also the run-dir stems).
_MODE = "enriched_rerank"
_STEMS = [
    "1804_03_21_1804032150",   # Code Civil
    "1867_06_08_1867060850",   # Code Pénal
    "1967_10_10_1967101055",   # Code judiciaire
    "1967_10_10_1967101056",   # Code judiciaire
    "2003_07_17_2013A31614",   # Code du Logement
]

_METHOD = "2C-enriched-rerank"


def _build_tree_maps(tree_root: dict) -> tuple[dict[int, dict], dict[str, dict]]:
    """Walk the deep tree once; return (bsard_id -> leaf, node_id -> node)."""
    bsard_to_node: dict[int, dict] = {}
    node_by_id: dict[str, dict] = {}

    def walk(node: dict) -> None:
        node_id = node.get("node_id")
        if node_id:
            node_by_id[str(node_id)] = node
        bs = node.get("bsard_id")
        if bs is not None:
            bsard_to_node[int(bs)] = node
        for child in node.get("sub_nodes") or []:
            walk(child)

    walk(tree_root)
    return bsard_to_node, node_by_id


def _normalise_query(q: dict, bsard_to_node: dict[int, dict]) -> dict:
    """Turn one Arm2C q<qid>.json into a RetrievalResult-shaped dict."""
    ranked_bsard_ids = q.get("ranked_bsard_ids") or []
    ranked_items: list[dict] = []
    for rank, bs in enumerate(ranked_bsard_ids, start=1):
        try:
            bs_int = int(bs)
        except (TypeError, ValueError):
            continue
        leaf = bsard_to_node.get(bs_int)
        # ``bsard_id`` (singular) lets the viewer's region-adapter resolve this
        # item to a PDF span via its bsard fallback (region_adapter.py:155-162),
        # exactly like T05 items. ``bsard_ids`` (plural, top-level) drives
        # GT-overlap classification + the sidebar lookups.
        meta: dict = {"source": "2C", "selected": False, "bsard_id": bs_int}
        if leaf is not None:
            node_id = leaf.get("node_id")
            if node_id:
                meta["node_id"] = node_id
            if leaf.get("text"):
                meta["text"] = leaf["text"]
        ranked_items.append(
            {
                "id": f"art_{bs_int}",
                "bsard_ids": [bs_int],
                "rank": rank,
                "score": None,
                "metadata": meta,
            }
        )

    steps = q.get("steps") or []
    visited_node_ids: list[str] = []
    selected_node_ids: list[str] = []
    for step in steps:
        nid = step.get("node_id")
        if nid:
            visited_node_ids.append(nid)
        for sec in step.get("sections") or []:
            if sec:
                visited_node_ids.append(sec)
        for art in step.get("articles") or []:
            if art:
                selected_node_ids.append(art)

    return {
        "query_id": str(q.get("query_id", "")),
        "query_text": str(q.get("query_text", "")),
        "method": _METHOD,
        "cost": {},
        "ranked_items": ranked_items,
        "trace": {
            "steps": steps,
            "summary": {
                "exit_reason": q.get("exit_reason"),
                "nodes_visited": q.get("nodes_visited"),
                "llm_calls": q.get("llm_calls"),
                "mode": "enriched-rerank",
                "visited_node_ids": visited_node_ids,
                "selected_node_ids": selected_node_ids,
                "selected_bsard_ids": list(q.get("selected_bsard_ids") or []),
                "ranked_bsard_ids_prererank": list(
                    q.get("ranked_bsard_ids_prererank") or []
                ),
            },
        },
    }


def export_stem(stem: str, out_root: Path) -> int:
    """Export one PDF stem's enriched_rerank run. Returns #queries written."""
    run_dir = _ARM2C_ROOT / "runs" / f"arm2c_{stem}_{_MODE}"
    tree_path = _ARM2C_ROOT / "data" / stem / "deep_tree.json"
    if not run_dir.is_dir():
        print(f"  [skip] no run dir: {run_dir}")
        return 0
    if not tree_path.exists():
        print(f"  [skip] no deep_tree.json: {tree_path}")
        return 0

    tree_data = json.loads(tree_path.read_text(encoding="utf-8"))
    bsard_to_node, _node_by_id = _build_tree_maps(tree_data.get("tree") or {})

    out_doc = out_root / stem
    out_results = out_doc / "results"
    out_results.mkdir(parents=True, exist_ok=True)

    n = 0
    for q_path in sorted(run_dir.glob("q*.json")):
        try:
            q = json.loads(q_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  [warn] could not read {q_path.name}: {exc}")
            continue
        normalised = _normalise_query(q, bsard_to_node)
        (out_results / q_path.name).write_text(
            json.dumps(normalised, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        n += 1

    # Copy the deep tree alongside so T08 can render the explorer.
    (out_doc / "deep_tree.json").write_text(
        json.dumps(tree_data, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  {stem}: {n} queries -> {out_results}")
    return n


def main() -> None:
    out_root = _T04_ROOT / "data" / "_arm2c"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Exporting Arm2C ({_MODE}) -> {out_root}")
    total = 0
    for stem in _STEMS:
        total += export_stem(stem, out_root)
    print(f"Done. {total} queries across {len(_STEMS)} PDFs.")


if __name__ == "__main__":
    main()
