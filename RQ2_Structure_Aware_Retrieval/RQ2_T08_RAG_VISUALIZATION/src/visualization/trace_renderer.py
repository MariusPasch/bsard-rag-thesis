"""Format PageIndex (2B) traces and GraphRAG (2C) metadata for side panel display."""

from __future__ import annotations


# ── PageIndex trace rendering ─────────────────────────────────────────────────


def format_pageindex_trace(trace: dict | None) -> str:
    """Render a PageIndex trace dict as a readable markdown summary.

    Trace shape produced by ``arm2_pageindex.retriever.state_to_retrieval_result``:

    .. code-block:: python

        {
          "summary": {
            "exit_reason": "no_candidates" | "sufficient" | "max_iterations",
            "iteration": int,
            "selected_law_ids":     [str, ...],
            "selected_chapter_ids": [str, ...],
            "candidate_article_ids":[str, ...],
            "n_candidates": int,
          },
          "steps": [
            # Per LLM call, in order:
            {
              "step": "law_selection" | "chapter_selection[<law_id>]"
                       | "article_selection[<chapter_id>]" | "evaluate" | ...,
              "iteration": int,
              "skipped": bool,         # present when the step was a no-op
              "reason": str,           # only when skipped
              "parsed": dict,          # the parsed JSON response from the LLM
              "parse_ok": bool,
              "tokens_in": int,
              "tokens_out": int,
              "latency_ms": float,
              "prompt_chars": int,     # length of the prompt sent to the LLM
            },
            ...
          ],
        }
    """
    if not trace:
        return "_No trace available._"

    summary = trace.get("summary") or {}
    steps = trace.get("steps") or []

    lines: list[str] = []

    # ── Summary block ────────────────────────────────────────────────────
    exit_reason = summary.get("exit_reason")
    iteration = summary.get("iteration")
    n_candidates = summary.get("n_candidates")
    if exit_reason or iteration is not None or n_candidates is not None:
        bits: list[str] = []
        if exit_reason:
            bits.append(f"**Exit:** `{exit_reason}`")
        if iteration is not None:
            bits.append(f"**Iter:** {iteration}")
        if n_candidates is not None:
            bits.append(f"**Candidates:** {n_candidates}")
        lines.append(" · ".join(bits))

    sel_laws = summary.get("selected_law_ids") or []
    if sel_laws:
        lines.append("**Laws:** " + ", ".join(f"`{x}`" for x in sel_laws))
    sel_chapters = summary.get("selected_chapter_ids") or []
    if sel_chapters:
        lines.append(
            "**Chapters:** "
            + ", ".join(f"`{x}`" for x in sel_chapters[:6])
            + (f" … (+{len(sel_chapters) - 6})" if len(sel_chapters) > 6 else "")
        )
    final_arts = summary.get("candidate_article_ids") or []
    if final_arts:
        lines.append(
            "**Final articles:** "
            + ", ".join(f"`{x}`" for x in final_arts[:8])
            + (f" … (+{len(final_arts) - 8})" if len(final_arts) > 8 else "")
        )

    # ── Per-step log ─────────────────────────────────────────────────────
    if steps:
        lines.append("")
        lines.append("**Steps**")
        for s in steps:
            name = s.get("step", "?")
            it = s.get("iteration")
            it_marker = f" `iter={it}`" if it not in (None, 0) else ""
            if s.get("skipped"):
                reason = s.get("reason", "")
                lines.append(f"- _skip_  `{name}`{it_marker} — {reason}")
                continue
            ok = "✓" if s.get("parse_ok") else "✗"
            tin = s.get("tokens_in", 0)
            tout = s.get("tokens_out", 0)
            ms = s.get("latency_ms")
            ms_str = f" · {int(ms)} ms" if ms else ""
            lines.append(
                f"- {ok}  `{name}`{it_marker} · "
                f"in={tin} → out={tout}{ms_str}"
            )

            parsed = s.get("parsed") or {}
            # Render the most informative fields per step type without
            # hard-coding them all — surface lists of selected ids and any
            # ``reason``/``status`` keys.
            for key, val in parsed.items():
                if isinstance(val, list):
                    if not val:
                        continue
                    head = ", ".join(f"`{v}`" for v in val[:5])
                    tail = f" … (+{len(val) - 5})" if len(val) > 5 else ""
                    lines.append(f"    - **{key}:** {head}{tail}")
                elif isinstance(val, str) and val:
                    snippet = val if len(val) <= 220 else val[:217] + "…"
                    lines.append(f"    - **{key}:** {snippet}")

    return "\n".join(lines) if lines else "_Trace is empty._"


# ── T04 (Arm 2A metadata) trace rendering ───────────────────────────────────


def format_t04_trace(trace: dict | None) -> str:
    """Render a T04 retrieval trace dict for the right-hand side panel.

    Trace fields produced by arm2_metadata pipeline (see compare_*.jsonl):
        variant, unit, sparse_only, boost_applied,
        n_terms_matched, jurisdiction_signal, article_refs,
        doc_id, config_hash
    """
    if not trace:
        return "_No trace available._"

    lines: list[str] = []

    variant = trace.get("variant")
    unit = trace.get("unit")
    if variant or unit:
        v = f"`{variant}`" if variant else "—"
        u = f"`{unit}`" if unit else "—"
        lines.append(f"**Variant:** {v} · **Unit:** {u}")

    flags: list[str] = []
    if trace.get("boost_applied"):
        flags.append("boosts ✓")
    if trace.get("sparse_only"):
        flags.append("sparse only")
    if flags:
        lines.append("**Flags:** " + " · ".join(flags))

    n_terms = trace.get("n_terms_matched")
    if n_terms is not None:
        lines.append(f"**Terms matched:** {n_terms}")

    jurisdiction = trace.get("jurisdiction_signal")
    if jurisdiction:
        lines.append(f"**Jurisdiction signal:** `{jurisdiction}`")

    refs = trace.get("article_refs") or []
    if refs:
        lines.append(
            "**Article refs in query:** "
            + ", ".join(f"`{r}`" for r in refs)
        )

    cfg = trace.get("config_hash")
    if cfg:
        lines.append(f"**Config:** `{cfg}`")

    return "\n\n".join(lines) if lines else "_Empty trace._"


# ── Arm2C (agentic ReAct tree-navigation) trace rendering ────────────────────


def format_arm2c_trace(trace: dict | None) -> str:
    """Render an Arm2C ReAct navigation trace for the side panel.

    Trace shape produced by ``export_for_t08._normalise_query``:

    .. code-block:: python

        {
          "summary": {
            "exit_reason": "frontier_empty" | "budget",
            "nodes_visited": int,
            "llm_calls": int,
            "mode": "enriched-rerank",
            "visited_node_ids":  [str, ...],
            "selected_node_ids": [str, ...],
            "selected_bsard_ids":[int, ...],
            "ranked_bsard_ids_prererank": [int, ...],
          },
          "steps": [
            {
              "node_id": str, "title": str, "n_children": int,
              "sections": [str, ...],   # child internal nodes descended into
              "articles": [str, ...],   # child leaves selected this step
              "thought": str,           # LLM reasoning (truncated)
              "parse_ok": bool, "latency_ms": float,
            }, ...
          ],
        }
    """
    if not trace:
        return "_No trace available._"

    summary = trace.get("summary") or {}
    steps = trace.get("steps") or []

    lines: list[str] = []

    # ── Summary block ────────────────────────────────────────────────────
    bits: list[str] = []
    exit_reason = summary.get("exit_reason")
    if exit_reason:
        bits.append(f"**Exit:** `{exit_reason}`")
    nodes_visited = summary.get("nodes_visited")
    if nodes_visited is not None:
        bits.append(f"**Nodes:** {nodes_visited}")
    llm_calls = summary.get("llm_calls")
    if llm_calls is not None:
        bits.append(f"**LLM calls:** {llm_calls}")
    if bits:
        lines.append(" · ".join(bits))

    # ── Per-step log ─────────────────────────────────────────────────────
    if steps:
        lines.append("")
        lines.append("**Navigation steps**")
        for i, s in enumerate(steps, start=1):
            ok = "✓" if s.get("parse_ok", True) else "✗"
            title = s.get("title") or s.get("node_id", "?")
            ms = s.get("latency_ms")
            ms_str = f" · {int(ms)} ms" if ms else ""
            lines.append(f"- {ok} **{i}.** {_truncate(title, 70)}{ms_str}")

            thought = (s.get("thought") or "").strip()
            if thought:
                lines.append(f"    - _{_truncate(thought, 200)}_")

            sections = s.get("sections") or []
            if sections:
                head = ", ".join(f"`{x}`" for x in sections[:4])
                tail = f" … (+{len(sections) - 4})" if len(sections) > 4 else ""
                lines.append(f"    - ↳ descended: {head}{tail}")

            articles = s.get("articles") or []
            if articles:
                head = ", ".join(f"`{x}`" for x in articles[:4])
                tail = f" … (+{len(articles) - 4})" if len(articles) > 4 else ""
                lines.append(f"    - ✓ selected: {head}{tail}")

    return "\n".join(lines) if lines else "_Trace is empty._"


def _truncate(s: str, limit: int = 90) -> str:
    s = (s or "").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


# ── Generic retrieval trace rendering ────────────────────────────────────────


def format_trace(result_trace: dict | None, method: str) -> str:
    """Dispatch to the correct trace renderer based on method prefix."""
    if not result_trace:
        return ""
    if method.startswith("2B") or method.startswith("arm2_pageindex"):
        return format_pageindex_trace(result_trace)
    if method.startswith("2A") or method.startswith("arm2_metadata"):
        return format_t04_trace(result_trace)
    if method.startswith("2C"):
        return format_arm2c_trace(result_trace)
    # Generic: pretty-print the dict
    import json
    try:
        return f"```json\n{json.dumps(result_trace, indent=2, default=str)}\n```"
    except Exception:
        return str(result_trace)
