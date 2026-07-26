"""End-to-end smoke for the navigator.

Two modes:
  ``--mode stub``  (default)  - uses a deterministic stub LLM that
                                returns canned JSON keyed on prompt
                                content. Validates plumbing without
                                requiring Ollama.
  ``--mode real``             - uses ``shared.llm.LLMClient``. Fails
                                fast if Ollama is not reachable on
                                localhost:11434.

Usage:
    python scripts/smoke_navigator.py
    python scripts/smoke_navigator.py --mode real --query "..." --tree ...
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict
from pathlib import Path

from arm2_pageindex.navigator import NavigatorConfig, navigate
from arm2_pageindex.tree_builder import load_law_tree


# ── Stub LLM ──────────────────────────────────────────────────────────────────


class _StubLLMResponse:
    __slots__ = ("text", "input_tokens", "output_tokens", "latency_ms")

    def __init__(self, text, input_tokens, output_tokens, latency_ms):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms


class StubLLMClient:
    """Returns canned JSON keyed on prompt content. No Ollama dependency
    and no import from ``shared`` so this works in any venv."""

    _LAW_RE = re.compile(r'"LAW_(\d+)"')
    _CHAPTER_RE = re.compile(r'"(LAW_\d+_CH_[^"]+)"')
    _ARTICLE_RE = re.compile(r'"(ART_doc_\d+_bsard_\d+)"')
    _ARTICLE_BRACKET_RE = re.compile(r'\[(ART_doc_\d+_bsard_\d+)\]')

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, system_prompt: str | None = None):
        self.calls += 1
        text = self._route(prompt)
        return _StubLLMResponse(
            text=text,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            latency_ms=5.0,
        )

    def generate_json(self, prompt: str, system_prompt: str | None = None):
        return json.loads(self._route(prompt))

    def _route(self, prompt: str) -> str:
        if "selected_law_ids" in prompt:
            m = self._LAW_RE.search(prompt)
            law_id = f"LAW_{m.group(1)}" if m else "LAW_2"
            return json.dumps({"selected_law_ids": [law_id], "reason": "stub"})
        if "selected_chapter_ids" in prompt:
            ids = self._CHAPTER_RE.findall(prompt)
            return json.dumps(
                {"selected_chapter_ids": ids[:2], "reason": "stub"}
            )
        if "selected_article_ids" in prompt:
            ids = self._ARTICLE_RE.findall(prompt)
            return json.dumps(
                {"selected_article_ids": ids[:3], "reason": "stub"}
            )
        if '"status": "sufficient"' in prompt or "Notez chaque article" in prompt:
            ids = self._ARTICLE_BRACKET_RE.findall(prompt)
            scores = {aid: (3 if i == 0 else 2) for i, aid in enumerate(ids)}
            return json.dumps({
                "status": "sufficient",
                "scores": scores,
                "additional_article_numbers": [],
                "reason": "stub",
            })
        return "{}"


# ── Smoke runner ──────────────────────────────────────────────────────────────


def _probe_ollama(base_url: str = "http://localhost:11434") -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    here = Path(__file__).resolve().parent
    project_root = here.parent

    parser = argparse.ArgumentParser(prog="smoke_navigator")
    parser.add_argument(
        "--tree", type=Path,
        default=project_root / "data" / "trees" / "doc_2.json",
        help="Path to a saved law-tree JSON.",
    )
    parser.add_argument(
        "--query", type=str,
        default=(
            "Quels sont les principes fondamentaux du droit de "
            "l'environnement en Région wallonne ?"
        ),
    )
    parser.add_argument("--query-id", default="smoke-q1")
    parser.add_argument("--mode", choices=("stub", "real"), default="stub")
    parser.add_argument(
        "--max-iterations", type=int, default=2,
        help="Cap on evaluate-loop iterations.",
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama base URL (real mode only).",
    )
    parser.add_argument(
        "--model", default="llama3.1:8b",
        help="Ollama model name (real mode only).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s - %(message)s",
    )
    log = logging.getLogger("smoke_navigator")

    if not args.tree.exists():
        log.error(
            "Tree not found at %s. Run scripts/smoke_tree.py first "
            "(or pass --tree).", args.tree,
        )
        return 2

    tree, manifest = load_law_tree(args.tree)
    log.info(
        "Loaded tree: doc_id=%s n_chapters=%s n_articles=%s "
        "chapter_derivable=%s",
        manifest.get("document_id"),
        manifest.get("n_chapters"),
        manifest.get("n_articles"),
        manifest.get("chapter_derivable"),
    )

    if args.mode == "real":
        if not _probe_ollama(args.ollama_url):
            log.error(
                "Ollama not reachable at %s. Start `ollama serve` and "
                "`ollama pull %s`, or use --mode stub.",
                args.ollama_url, args.model,
            )
            return 3
        from shared.llm import LLMClient
        llm = LLMClient(model=args.model, base_url=args.ollama_url)
        log.info("Using real LLM: %s @ %s", args.model, args.ollama_url)
    else:
        llm = StubLLMClient()
        log.info("Using stub LLM (no Ollama)")

    cfg = NavigatorConfig(max_iterations=args.max_iterations)
    state = navigate(
        query=args.query,
        query_id=args.query_id,
        tree=tree,
        llm=llm,
        cfg=cfg,
    )

    print()
    print("=" * 78)
    print(f"Query: {args.query}")
    print(f"Mode:  {args.mode}")
    print("=" * 78)
    print(f"Exit reason:           {state.exit_reason}")
    print(f"Iterations:            {state.iteration}")
    print(f"LLM calls:             {state.cost['llm_calls']}")
    print(f"Tokens in / out:       {state.cost['tokens_in']} / {state.cost['tokens_out']}")
    print(f"Latency total:         {state.cost['latency_ms']:.1f} ms")
    print(f"Parse failures:        {state.cost['parse_failures']}")
    print(f"Selected laws:         {state.selected_law_ids}")
    print(f"Selected chapters:     {len(state.selected_chapter_ids)}")
    print(f"Candidate articles:    {len(state.candidate_article_ids)}")
    print(f"Final ranked articles: {len(state.final_ranked)}")
    print()
    print("Top-5 ranked:")
    for aid, score in state.final_ranked[:5]:
        print(f"  score={score:>4.1f}  {aid}")
    print()
    print("Trace summary (per step):")
    for entry in state.trace:
        step = entry.get("step", "?")
        if entry.get("skipped"):
            print(f"  [SKIP]   {step:<35}  reason={entry.get('reason')}")
        else:
            print(
                f"  parse={'ok' if entry.get('parse_ok') else 'FAIL'}  "
                f"{step:<35}  in={entry.get('tokens_in', 0):>5}  "
                f"out={entry.get('tokens_out', 0):>5}  "
                f"{entry.get('latency_ms', 0.0):>7.1f} ms"
            )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
