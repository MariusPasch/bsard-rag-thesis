"""Tier (b) acceptance gate — control-flow test for navigate_corrective() with a
scripted MockLLM. No Ollama, no model. Proves the loop's mechanics:

  1. memory carries across rounds (visited accumulates round 1 + round 2 nodes)
  2. the corrective re-seed PULLS the section pruned in round 1 into round 2
  3. round 1 alone MISSES the gold; the corrective round is what rescues it
  4. the union reached pool grows monotonically across rounds
  5. the loop terminates on the round cap
  6. a parse failure mid-descent does not crash the run
  7. reseed_strategy="all" takes the no-LLM brute-force path

Run:  python corrective/test_corrective_mock.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))                 # corrective/
sys.path.insert(0, str(_HERE.parent.parent))         # experiments/arm2c
from navigator_tools import DeepTree, Node            # noqa: E402
from react_navigator import SYSTEM_PROMPT, RERANK_SYSTEM  # noqa: E402
from corrective_navigator import (                    # noqa: E402
    navigate_corrective, RESEED_SYSTEM, _rank_pruned,
)


# ── synthetic tree ───────────────────────────────────────────────────────────
#   LAW (1 child)
#     ROOT (multiway, 2 top branches)
#       BRANCH_A  -> leaf A1, leaf A2        (irrelevant; the agent descends here)
#       BRANCH_B  -> SUB_B -> leaf GOLD(777), leaf B2   (relevant; pruned round 1)
GOLD = 777

def _leaf(nid, bsard):
    return Node(node_id=nid, title=nid, article_id=f"art_{bsard}", bsard_id=bsard,
                text=f"text {bsard}")

def _build_tree() -> DeepTree:
    a1, a2 = _leaf("ART_A1_bsard_101", 101), _leaf("ART_A2_bsard_102", 102)
    gold, b2 = _leaf("ART_GOLD_bsard_777", GOLD), _leaf("ART_B2_bsard_103", 103)
    branch_a = Node(node_id="BRANCH_A", title="BRANCH_A", sub_nodes=[a1, a2])
    sub_b = Node(node_id="SUB_B", title="SUB_B", sub_nodes=[gold, b2])
    branch_b = Node(node_id="BRANCH_B", title="BRANCH_B", sub_nodes=[sub_b])
    root = Node(node_id="ROOT", title="ROOT", sub_nodes=[branch_a, branch_b])
    law = Node(node_id="LAW", title="LAW", sub_nodes=[root])
    return DeepTree(law)


# ── scripted mock LLM ────────────────────────────────────────────────────────

class MockLLM:
    """Returns deterministic JSON keyed by which prompt it sees. `prune_b` toggles
    whether round-1 descends BRANCH_B (False = prune it, the interesting case).
    `raise_on` raises once when the given title is the current descent node."""

    def __init__(self, prune_b: bool = False, raise_on: str | None = None):
        self.prune_b = prune_b
        self.raise_on = raise_on
        self.calls: list[str] = []          # log of (kind:title) for assertions

    def generate_json(self, prompt: str, system_prompt: str | None = None) -> dict:
        if system_prompt == RESEED_SYSTEM:
            self.calls.append("reseed")
            return {"ranking": [1]}          # re-explore the (single) pruned section
        if system_prompt == RERANK_SYSTEM:
            self.calls.append("rerank")
            return {"ranking": [1]}          # identity-ish; rest keep nav order
        # otherwise it's a descent call — find the current node title
        title = prompt.split("Section actuelle : ", 1)[1].split("\n", 1)[0].strip()
        self.calls.append(f"descend:{title}")
        if self.raise_on and title == self.raise_on:
            self.raise_on = None
            raise RuntimeError("injected parse failure")
        if title == "LAW":
            return {"sections": [1], "articles": []}             # -> ROOT
        if title == "ROOT":
            return {"sections": ([1, 2] if self.prune_b else [1]),  # [1]=A only => prune B
                    "articles": []}
        if title == "BRANCH_A":
            return {"sections": [], "articles": []}              # nothing useful
        if title == "BRANCH_B":
            return {"sections": [1], "articles": []}             # -> SUB_B
        if title == "SUB_B":
            return {"sections": [], "articles": [1]}             # select GOLD ([1])
        return {"sections": [], "articles": []}


def _ok(cond, msg):
    print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
    assert cond, msg


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    fails = 0

    # ── 1: single pass (max_rounds=1) MISSES the gold ────────────────────────
    print("[1] single-pass control (max_rounds=1): gold should be MISSED")
    tree = _build_tree()
    llm = MockLLM(prune_b=False)
    r1 = navigate_corrective("q", tree, llm, mode="bare", max_rounds=1,
                             reseed_strategy="ranked", rerank=False)
    _ok(GOLD not in r1.selected_bsard_ids, "round-1-only does not select the gold")
    _ok("BRANCH_B" not in {s.node_id for s in r1.steps}, "round 1 pruned BRANCH_B")
    _ok(len(r1.rounds) == 1 and r1.rounds[0].reseed_call is False,
        "no re-seed call when max_rounds=1")

    # ── 2: corrective loop (max_rounds=2) RESCUES the gold ───────────────────
    print("\n[2] corrective loop (max_rounds=2): re-seed should rescue the gold")
    tree = _build_tree()
    llm = MockLLM(prune_b=False)
    r2 = navigate_corrective("q", tree, llm, mode="bare", max_rounds=2,
                             reseed_strategy="ranked", rerank=False)
    visited = {s.node_id for s in r2.steps}
    _ok({"LAW", "ROOT", "BRANCH_A"} <= visited, "memory carries round-1 nodes")
    _ok({"BRANCH_B", "SUB_B"} <= visited, "re-seed pulled the pruned BRANCH_B + descended it")
    _ok(GOLD in r2.selected_bsard_ids, "corrective round selects the gold")
    _ok(any(c == "reseed" for c in llm.calls), "a re-seed ranking call was made")
    reach = [ri.reach_total for ri in r2.rounds]
    _ok(reach == sorted(reach) and reach[-1] > reach[0],
        f"reached pool grows monotonically across rounds: {reach}")
    _ok(len(r2.rounds) == 2 and r2.exit_reason in ("rounds_done", "no_growth"),
        f"terminates on the round cap (exit={r2.exit_reason})")

    # ── 3: termination — no infinite loop even with a big cap ─────────────────
    print("\n[3] termination: large cap must still stop (no-growth / no-pruned-left)")
    tree = _build_tree()
    llm = MockLLM(prune_b=False)
    r3 = navigate_corrective("q", tree, llm, mode="bare", max_rounds=10,
                             reseed_strategy="ranked", rerank=False)
    _ok(len(r3.rounds) <= 10 and r3.exit_reason in ("no_growth", "no_pruned_left", "rounds_done"),
        f"loop halts in {len(r3.rounds)} rounds (exit={r3.exit_reason})")
    _ok(GOLD in r3.selected_bsard_ids, "gold still recovered under the large cap")

    # ── 4: parse failure mid-descent does not crash ──────────────────────────
    print("\n[4] robustness: an injected parse failure must not crash the run")
    tree = _build_tree()
    llm = MockLLM(prune_b=False, raise_on="BRANCH_A")
    try:
        r4 = navigate_corrective("q", tree, llm, mode="bare", max_rounds=2,
                                 reseed_strategy="ranked", rerank=False)
        crashed = False
    except Exception as exc:                                   # noqa: BLE001
        crashed = True
        print(f"     crashed: {exc}")
    _ok(not crashed, "run completes despite a parse failure")
    _ok(any(not s.parse_ok for s in r4.steps), "the parse failure is recorded, not silent")
    _ok(GOLD in r4.selected_bsard_ids, "gold still recovered (failure was on the irrelevant branch)")

    # ── 5: reseed_strategy='all' takes the no-LLM brute-force path ────────────
    print("\n[5] reseed_strategy='all': brute-force re-seed makes NO ranking call")
    seed, made_call = _rank_pruned("q", ["BRANCH_B", "X", "Y"], _build_tree(),
                                   None, "bare", 5, "all")
    _ok(seed == ["BRANCH_B", "X", "Y"] and made_call is False,
        "'all' returns candidates in order with no LLM call")

    print("\n=== ALL CONTROL-FLOW TESTS PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
