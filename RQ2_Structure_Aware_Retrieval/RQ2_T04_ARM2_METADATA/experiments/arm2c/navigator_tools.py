"""Arm-2C navigator substrate: the deep tree + four tools + preflight.

Pure, LLM-free, unit-testable. The ReAct loop (next module) sits on top of this
and only adds: prompt the model, parse its action, call these, repeat.

Tools (all keyed by canonical node_id):
  expand[id]  -> a section's children (branch decision)     [internal nodes only]
  open[id]    -> an article's full French text              [leaves only]
  select[id]  -> keep an article                            [leaves only]
  finish      -> stop

Preflight validates every (action, id) BEFORE it runs and returns a French error
string instead of throwing, so the loop can feed the model a correction (the
Tier42 lesson: never let a malformed action crash the run).

Rendering has two registers, toggled by `mode`:
  "bare"     -> French label + article count / French first-sentence
  "enriched" -> + native English level_summary (branches),
                  content_summary + keywords (leaves), tagged (EN)
Instruction language stays French (= query language); the multilingual LLM
bridges. See PROJECT memory: metadata-enrichment design.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve()


@dataclass
class Node:
    """Minimal deep_tree.json node — vendored so the Arm-2C runtime depends on
    nothing but stdlib (no arm2_pageindex / shared / rank_bm25 on the VM).
    Schema matches what deep_tree_builder.py writes (Node.to_dict)."""
    node_id: str
    title: str = ""
    summary: str = ""
    metadata: dict = field(default_factory=dict)
    sub_nodes: list = field(default_factory=list)
    article_id: str | None = None
    bsard_id: int | None = None
    text: str | None = None

    @property
    def is_leaf(self) -> bool:
        return self.article_id is not None

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        return cls(
            node_id=d["node_id"], title=d.get("title", ""), summary=d.get("summary", ""),
            metadata=d.get("metadata", {}),
            sub_nodes=[cls.from_dict(c) for c in d.get("sub_nodes", [])],
            article_id=d.get("article_id"), bsard_id=d.get("bsard_id"), text=d.get("text"),
        )


_BRACKET = re.compile(r"\[\s*\d+\s*|\s*\]\s*\d+")  # Belgian amendment markers [1 .. ]1


def clean_title(t: str) -> str:
    """Strip Belgian amendment brackets for cleaner prompts; cosmetic only."""
    return re.sub(r"\s+", " ", _BRACKET.sub(" ", t or "")).strip(" .-") or (t or "")


# ── tree wrapper ─────────────────────────────────────────────────────────────


class DeepTree:
    """Loaded deep_tree.json with O(1) id lookup + parent/child maps."""

    def __init__(self, root: Node, manifest: dict | None = None):
        self.root = root
        self.manifest = manifest or {}
        self.by_id: dict[str, Node] = {}
        self.parent: dict[str, str | None] = {}
        self._index(root, None)

    def _index(self, node: Node, parent_id: str | None) -> None:
        self.by_id[node.node_id] = node
        self.parent[node.node_id] = parent_id
        for c in node.sub_nodes:
            self._index(c, node.node_id)

    @classmethod
    def load(cls, path: str | Path) -> "DeepTree":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(Node.from_dict(payload["tree"]), payload.get("manifest", {}))

    def node(self, node_id: str) -> Node | None:
        return self.by_id.get(node_id)

    def is_leaf(self, node_id: str) -> bool:
        n = self.by_id.get(node_id)
        return bool(n and n.is_leaf)


# ── navigation state ─────────────────────────────────────────────────────────


@dataclass
class NavState:
    selected: list[str] = field(default_factory=list)   # leaf node_ids, ordered
    visited: set[str] = field(default_factory=set)      # expanded node_ids
    steps: int = 0
    finished: bool = False

    def select(self, node_id: str) -> None:
        if node_id not in self.selected:
            self.selected.append(node_id)

    def selected_bsard_ids(self, tree: DeepTree) -> list[int]:
        out, seen = [], set()
        for nid in self.selected:
            n = tree.node(nid)
            if n and n.bsard_id is not None and n.bsard_id not in seen:
                seen.add(n.bsard_id)
                out.append(n.bsard_id)
        return out


# ── preflight ────────────────────────────────────────────────────────────────

VALID_ACTIONS = ("expand", "open", "select", "finish")


def preflight(tree: DeepTree, action: str, node_id: str | None) -> str | None:
    """Return a French error string if (action, node_id) is illegal, else None."""
    if action not in VALID_ACTIONS:
        return f"Action inconnue : « {action} ». Utilisez expand, open, select ou finish."
    if action == "finish":
        return None
    if not node_id:
        return f"L'action « {action} » exige un identifiant de nœud."
    n = tree.node(node_id)
    if n is None:
        return f"Nœud «{node_id}» introuvable."
    if action == "expand" and n.is_leaf:
        return f"«{node_id}» est un article, pas une section — utilisez open ou select."
    if action in ("open", "select") and not n.is_leaf:
        return f"«{node_id}» est une section, pas un article — utilisez expand pour l'explorer."
    return None


# ── tools (assume preflight already passed) ──────────────────────────────────


def _child_view(tree: DeepTree, c: Node, mode: str) -> dict:
    meta = c.metadata or {}
    if c.is_leaf:
        view = {"id": c.node_id, "kind": "article",
                "title": clean_title(c.title), "summary_fr": c.summary or ""}
        if mode == "enriched":
            if meta.get("content_summary_en"):
                view["summary_en"] = meta["content_summary_en"]
            if meta.get("keywords_en"):
                view["keywords_en"] = meta["keywords_en"]
    else:
        kind = "unfiled" if meta.get("orphan") else "section"
        view = {"id": c.node_id, "kind": kind,
                "title": clean_title(c.title),
                "article_count": meta.get("article_count", len(c.sub_nodes))}
        if mode == "enriched" and meta.get("level_summary_en"):
            view["summary_en"] = meta["level_summary_en"]
    return view


def expand(tree: DeepTree, node_id: str, mode: str = "bare") -> list[dict]:
    """Children of an internal node, as compact views for the prompt layer."""
    n = tree.node(node_id)
    return [_child_view(tree, c, mode) for c in n.sub_nodes]


def open_article(tree: DeepTree, node_id: str) -> dict:
    n = tree.node(node_id)
    return {"id": n.node_id, "title": clean_title(n.title),
            "bsard_id": n.bsard_id, "text": n.text or ""}


def select(state: NavState, tree: DeepTree, node_id: str) -> None:
    state.select(node_id)


# ── prompt rendering (numbered menu + number->id map for the LLM) ────────────


def _crop(s: str, n: int | None) -> str:
    if not n or len(s) <= n:
        return s
    return s[:n].rsplit(" ", 1)[0] + "…"


def render_children(views: list[dict], mode: str = "bare",
                    summary_chars: int | None = None) -> tuple[str, dict[int, str]]:
    """Numbered French menu the LLM picks from; returns (text, {n: node_id}).
    `summary_chars` caps enriched (EN) summaries to keep 8B prompt context small."""
    lines, num2id = [], {}
    for i, v in enumerate(views, 1):
        num2id[i] = v["id"]
        if v["kind"] == "article":
            head = f"  [{i}] {v['title']}"
            if mode == "enriched" and v.get("summary_en"):
                head += f" — (EN) {_crop(v['summary_en'], summary_chars)}"
                if v.get("keywords_en"):
                    head += f"  · mots-clés (EN): {v['keywords_en']}"
            else:
                head += f" — {v.get('summary_fr','')[:90]}"
            lines.append(head)
        else:
            tag = "  [Non rattaché]" if v["kind"] == "unfiled" else ""
            head = f"  [{i}] {v['title']}  ({v['article_count']} art.){tag}"
            if mode == "enriched" and v.get("summary_en"):
                head += f"\n      (EN) {_crop(v['summary_en'], summary_chars)}"
            lines.append(head)
    return "\n".join(lines), num2id


# ── self-test: scripted descent, no LLM ──────────────────────────────────────


def _selftest(tree_path: Path) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # French titles on cp1252 consoles
    except Exception:
        pass
    tree = DeepTree.load(tree_path)
    print(f"loaded {tree_path.name}: {len(tree.by_id)} nodes, "
          f"root={tree.root.node_id!r} ({tree.root.title!r})\n")

    # preflight rejections
    assert preflight(tree, "expand", "NOPE") and "introuvable" in preflight(tree, "expand", "NOPE")
    a_leaf = next(n for n in tree.by_id.values() if n.is_leaf)
    assert preflight(tree, "expand", a_leaf.node_id), "expand on leaf should fail"
    assert preflight(tree, "open", tree.root.node_id), "open on root should fail"
    assert preflight(tree, "expand", tree.root.node_id) is None
    print("preflight: rejects unknown id, expand-on-leaf, open-on-section ✓\n")

    # scripted descent: root -> first real section -> ... -> first leaf, in both modes
    for mode in ("bare", "enriched"):
        print(f"===== mode={mode} =====")
        cur = tree.root.node_id
        for _ in range(8):
            views = expand(tree, cur, mode=mode)
            text, num2id = render_children(views, mode=mode)
            print(f"expand[{cur}] -> {tree.node(cur).title[:45]!r}")
            print(text[:600])
            # descend into the first non-leaf, non-unfiled child; else stop at leaves
            nxt = next((v["id"] for v in views if v["kind"] == "section"), None)
            if nxt is None:
                first_leaf = next((v["id"] for v in views if v["kind"] == "article"), None)
                if first_leaf:
                    art = open_article(tree, first_leaf)
                    print(f"\nopen[{first_leaf}] -> {art['title']!r} bsard={art['bsard_id']}")
                    print(f"  text: {art['text'][:120]!r}")
                    st = NavState()
                    select(st, tree, first_leaf)
                    print(f"select -> bsard_ids={st.selected_bsard_ids(tree)}")
                break
            cur = nxt
            print()
        print()
    print("self-test OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path,
                    default=_HERE.parent / "data" / "1804_03_21_1804032150" / "deep_tree.json")
    args = ap.parse_args()
    return _selftest(args.tree)


if __name__ == "__main__":
    sys.exit(main())
