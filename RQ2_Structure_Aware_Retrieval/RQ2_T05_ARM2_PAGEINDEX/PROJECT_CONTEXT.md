# 05 — ARM 2B PAGEINDEX CONTEXT
## Vectorless Hierarchy Reasoning via ToC Tree Navigation

---

## 1. PURPOSE

A vectorless, reasoning-based retrieval method. Instead of embedding and similarity search, the LLM navigates a hierarchical Table-of-Contents tree built from the document's structure. The LLM reasons about which branch of the hierarchy is relevant, drills down to specific articles, and follows cross-references — mimicking how a legal expert navigates a code of law.

Reference: Zhang & Tang (2025), "PageIndex: Next-Generation Vectorless, Reasoning-based RAG", PageIndex Blog.

## 2. DIRECTORY STRUCTURE

```
arm2_pageindex/
├── __init__.py
├── tree_builder.py      # Deterministic ToC tree: build_law_tree + compose_corpus_tree (ToCNode)
├── navigator.py         # Per-query nav loop navigate() over NavigationState (plain Python, no LangGraph)
├── prompts.py           # French JSON-strict prompt templates for each navigation step
├── retriever.py         # run_arm2b(): wraps navigate() -> RetrievalResult
└── pipeline.py          # run_subset(): batch runner with per-query JSON cache
```

## 3. DEPENDENCIES

- Uses: `shared.llm` (LLaMA 3.1 8B via Ollama); T04's `bsard_link` for node→bsard_id;
  T04 transitively pulls in T03's `arm1_naive.chunker`
- Input: `AzureNode` + `Article` units (same AzureDI-derived corpus as T04), built into a
  per-law `ToCNode` tree. It does **not** consume T02's `DocumentBundle.articles` directly.
- Output: `list[RetrievalResult]` (with full navigation `trace`)
- Does NOT use: `shared.embeddings`, `shared.faiss_store`, `shared.bm25_store` (vectorless)

## 4. TOC TREE CONSTRUCTION (tree_builder.py)

### 4.1 Tree Schema

```python
@dataclass
class ToCNode:
    node_id: str
    title: str
    summary: str
    metadata: dict
    sub_nodes: list['ToCNode'] = field(default_factory=list)
    content_ref: str | None = None    # article_id for leaf nodes → maps to full text
    
    def to_json(self, include_content: bool = False) -> dict:
        """Serialize for LLM prompt. Excludes content_ref unless requested."""
        d = {
            "node_id": self.node_id,
            "title": self.title,
            "summary": self.summary,
        }
        if self.metadata:
            d["metadata"] = self.metadata
        if self.sub_nodes:
            d["sub_nodes"] = [n.to_json(include_content) for n in self.sub_nodes]
        return d
    
    def to_level_json(self, max_depth: int = 1) -> dict:
        """Serialize only to a certain depth (for fitting in context window)."""
        d = {"node_id": self.node_id, "title": self.title, "summary": self.summary}
        if self.metadata:
            d["metadata"] = {k: v for k, v in self.metadata.items() 
                            if k in ('article_count', 'jurisdiction', 'defined_terms')}
        if max_depth > 0 and self.sub_nodes:
            d["sub_nodes"] = [n.to_level_json(max_depth - 1) for n in self.sub_nodes]
        elif self.sub_nodes:
            d["sub_node_count"] = len(self.sub_nodes)
        return d
```

### 4.2 Tree Construction

```python
# REAL API (tree_builder.py): per-law subtree, then compose a corpus root.
#   build_law_tree(*, doc_id, doc_meta, articles_for_doc, nodes_for_doc) -> (law_node, chapter_derivable)
#   compose_corpus_tree(law_subtrees: list[ToCNode]) -> ToCNode
#   save_law_tree(...) / load_law_tree(path) cache to data/trees/, pinning pdf_sha256 + TREE_BUILDER_VERSION.
# The block below is ILLUSTRATIVE of the Law -> Chapter -> Article shaping only.
def _illustrative_build(bundles):  # not the real signature
    """Build hierarchical ToC tree (Law -> Chapter -> Article). Leaf nodes carry
    content_ref -> full article text, loaded only when the LLM selects an article."""
    content_lookup = {}
    
    # Level 0: Corpus root
    root = ToCNode(
        node_id="ROOT",
        title="Belgian Statutory Corpus",
        summary=f"Contains {sum(len(b.articles) for b in bundles)} articles "
                f"across {len(bundles)} legislative documents",
        metadata={"document_count": len(bundles)}
    )
    
    for bundle in bundles:
        if not bundle.has_azure_extraction or not bundle.articles:
            continue
        
        meta = bundle.metadata
        
        # Level 1: Law node
        law_node = ToCNode(
            node_id=f"LAW_{bundle.document_id}",
            title=meta.get('DocumentTitle', f'Document {bundle.document_id}'),
            summary=first_sentence(meta.get('Summary', '')),
            metadata={
                'jurisdiction': meta.get('Jurisdiction', ''),
                'issuing_authority': meta.get('IssuingAuthority', ''),
                'document_number': meta.get('DocumentNumber', ''),
                'entry_into_force': meta.get('EntryIntoForceDate', ''),
                'article_count': len(bundle.articles),
                'defined_terms': bundle.defined_terms_list[:20],  # cap for context window
            }
        )
        
        # Level 2: Group articles by chapter
        chapters = group_articles_by_chapter(bundle.articles)
        
        for chapter_key, chapter_articles in chapters.items():
            chapter_node = ToCNode(
                node_id=f"LAW_{bundle.document_id}_CH_{sanitize_id(chapter_key)}",
                title=chapter_key if chapter_key else "General Provisions",
                summary=f"Contains {len(chapter_articles)} articles "
                        f"({chapter_articles[0].article_number} to "
                        f"{chapter_articles[-1].article_number})",
                metadata={'article_count': len(chapter_articles)}
            )
            
            # Level 3: Article leaf nodes
            for article in chapter_articles:
                article_node = ToCNode(
                    node_id=f"ART_{article.article_id}",
                    title=f"Article {article.article_number}" if article.article_number 
                          else f"Article {article.article_id}",
                    summary=first_sentence(article.text),
                    metadata={
                        'token_count': article.token_count,
                        'defined_terms_used': article.defined_terms[:5],  # cap
                        'cross_references': article.cross_references[:5],  # cap
                    },
                    content_ref=article.article_id
                )
                chapter_node.sub_nodes.append(article_node)
                content_lookup[article.article_id] = article.text
            
            law_node.sub_nodes.append(chapter_node)
        
        root.sub_nodes.append(law_node)
    
    return root, content_lookup

def group_articles_by_chapter(articles: list[Article]) -> dict[str, list[Article]]:
    """Group articles by their chapter field. Preserves order."""
    chapters = OrderedDict()
    for article in articles:
        key = article.chapter or "_ungrouped"
        if key not in chapters:
            chapters[key] = []
        chapters[key].append(article)
    return chapters
```

**Key design:** The tree is **deterministic** — no LLM calls needed. Belgian statutory law has an explicit hierarchy (Law → Chapter → Section → Article) already parsed by Azure DI. If chapter metadata is missing, use a flat two-level tree (Law → Articles).

### 4.3 Fallback for Missing Hierarchy

```python
if not any(a.chapter for a in bundle.articles):
    # No chapter metadata — flat structure
    # All articles become direct children of the law node
    for article in bundle.articles:
        law_node.sub_nodes.append(article_node)
    # Skip chapter level
```

## 5. LLM NAVIGATION (navigator.py)

### 5.1 State

> The real navigator uses a `NavigationState` **dataclass** (navigator.py) configured by
> `NavigatorConfig`, driven by a plain-Python `navigate(*, query, query_id, tree, llm, cfg)`
> loop — **not** the `TypedDict` / LangGraph sketch shown below (illustrative of the carried
> fields only).

```python
from typing import TypedDict

class PageIndexState(TypedDict):  # illustrative; real type is NavigationState (dataclass)
    query: str
    tree: ToCNode                       # Full tree reference
    content_lookup: dict                # article_id → text
    selected_laws: list[str]            # node_ids
    selected_chapters: list[str]        # node_ids
    selected_articles: list[str]        # node_ids
    retrieved_text: dict[str, str]      # article_id → full text
    iteration: int
    max_iterations: int
    final_article_ids: list[str]
    reasoning_trace: list[dict]         # Full prompt/response log
```

### 5.2 Navigation Steps

**Step 2A — Law Selection:**

Skip if only one document loaded (`skip_law_selection_if_single_doc` config).

```python
def select_laws(state: PageIndexState, llm: LLMClient) -> PageIndexState:
    level_1_json = json.dumps(
        [node.to_level_json(max_depth=0) for node in state['tree'].sub_nodes],
        indent=2, ensure_ascii=False
    )
    response = llm.generate_json(
        prompt=PROMPTS['law_selection'].format(
            query=state['query'],
            level_1_tree=level_1_json
        )
    )
    state['selected_laws'] = response.get('selected_law_ids', [])
    state['reasoning_trace'].append({'step': 'law_selection', 'response': response})
    return state
```

**Step 2B — Chapter Navigation:**

```python
def select_chapters(state: PageIndexState, llm: LLMClient) -> PageIndexState:
    all_selected = []
    for law_id in state['selected_laws']:
        law_node = find_node(state['tree'], law_id)
        level_2_json = json.dumps(
            [ch.to_level_json(max_depth=0) for ch in law_node.sub_nodes],
            indent=2, ensure_ascii=False
        )
        response = llm.generate_json(
            prompt=PROMPTS['chapter_selection'].format(
                query=state['query'],
                law_title=law_node.title,
                level_2_tree=level_2_json
            )
        )
        all_selected.extend(response.get('selected_chapter_ids', []))
    state['selected_chapters'] = all_selected
    state['reasoning_trace'].append({'step': 'chapter_selection', 'response': response})
    return state
```

**Step 2C — Article Selection:**

```python
def select_articles(state: PageIndexState, llm: LLMClient) -> PageIndexState:
    all_selected = []
    for chapter_id in state['selected_chapters']:
        chapter_node = find_node(state['tree'], chapter_id)
        level_3_json = json.dumps(
            [art.to_json() for art in chapter_node.sub_nodes],
            indent=2, ensure_ascii=False
        )
        response = llm.generate_json(
            prompt=PROMPTS['article_selection'].format(
                query=state['query'],
                chapter_title=chapter_node.title,
                level_3_nodes=level_3_json
            )
        )
        all_selected.extend(response.get('selected_article_ids', []))
    state['selected_articles'] = all_selected
    # Load full text for selected articles
    for art_id in all_selected:
        clean_id = art_id.replace("ART_", "")
        if clean_id in state['content_lookup']:
            state['retrieved_text'][clean_id] = state['content_lookup'][clean_id]
    return state
```

**Step 2D — Read, Evaluate, and Follow Cross-References:**

```python
def evaluate_and_follow_refs(state: PageIndexState, llm: LLMClient) -> PageIndexState:
    articles_text = "\n\n---\n\n".join(
        f"[{art_id}]\n{text}" for art_id, text in state['retrieved_text'].items()
    )
    response = llm.generate_json(
        prompt=PROMPTS['evaluate'].format(
            query=state['query'],
            article_texts=articles_text
        )
    )
    
    if response.get('status') == 'sufficient':
        state['final_article_ids'] = response['final_article_ids']
    elif response.get('status') == 'need_more':
        # Follow cross-references
        for ref_id in response.get('additional_articles', []):
            if ref_id in state['content_lookup'] and ref_id not in state['retrieved_text']:
                state['retrieved_text'][ref_id] = state['content_lookup'][ref_id]
        state['iteration'] += 1
    
    state['reasoning_trace'].append({'step': 'evaluate', 'response': response})
    return state
```

### 5.3 Navigation Loop (no LangGraph)

There is **no LangGraph dependency**. `navigate()` runs the steps as a plain Python loop:
law-selection (skipped for a single-law tree) → chapter-selection → article-selection →
evaluate; on `need_more` it follows cross-references and iterates up to
`NavigatorConfig.max_iterations`, else it exits. Each step logs prompt size, parsed
response, latency, and parse failures into the `NavigationState` trace; `exit_reason` and
`iterations` are recorded for T07's cost tracker. `retriever.run_arm2b` wraps a single
`navigate()` call and converts the final `NavigationState` into a `RetrievalResult` via
`state_to_retrieval_result`.

## 6. PROMPTS (prompts.py)

```python
PROMPTS = {
    "law_selection": """You are a legal retrieval expert. Given a legal question, select the most relevant law(s).

Question: "{query}"

Available laws:
{level_1_tree}

Select 1-3 laws most likely to contain the answer.
Respond as JSON only: {{"selected_law_ids": ["LAW_2", ...]}}""",

    "chapter_selection": """Question: "{query}"
Selected law: {law_title}

Chapters in this law:
{level_2_tree}

Select 1-5 chapters most likely to contain the answer.
Respond as JSON only: {{"selected_chapter_ids": ["LAW_2_CH_3", ...]}}""",

    "article_selection": """Question: "{query}"
Chapter: {chapter_title}

Articles in this chapter:
{level_3_nodes}

Select the articles relevant to the question.
Respond as JSON only: {{"selected_article_ids": ["ART_35", "ART_37", ...]}}""",

    "evaluate": """Question: "{query}"

Selected articles:
{article_texts}

1. Are these articles sufficient to answer the question?
2. Do any articles reference other articles that should be included?
   (Look for patterns like "visé à l'article D.49", "conformément à l'article...")

If sufficient, respond: {{"status": "sufficient", "final_article_ids": [...]}}
If more articles needed: {{"status": "need_more", "additional_articles": ["D.49", ...], "reason": "..."}}

Respond as JSON only."""
}
```

## 7. INTERFACE FOR ORCHESTRATOR

The real public entry points (both keyword-only) are in `retriever.py` / `pipeline.py`:

```python
def run_arm2b(*, query: str, query_id: str, tree: ToCNode, llm: LLMClient,
              cfg: NavigatorConfig | None = None,
              method_name: str = DEFAULT_METHOD_NAME) -> RetrievalResult:
    """One query over a pre-built tree -> RetrievalResult (ranked_items, cost, trace)."""

def run_subset(*, tree: ToCNode, queries: list[dict], llm: LLMClient,
               cfg: NavigatorConfig | None = None,
               out_dir: Path | str | None = None, ...) -> list[RetrievalResult]:
    """Batch runner over [{"query_id", "query_text"}, ...]; writes idempotent
    per-query JSON + manifest under out_dir."""

# Tree is built once (deterministic, no LLM) and cached:
#   law_node, _ = build_law_tree(doc_id=..., doc_meta=..., articles_for_doc=..., nodes_for_doc=...)
#   tree = compose_corpus_tree([law_node, ...]); save_law_tree(tree, path)
```

> **Note:** the T00 orchestrator still imports a legacy `run_pageindex(bundle, config)` name
> that does not exist in the shipped package; in practice T05 is run via `scripts/build_tree.py`
> + the Azure notebook (`notebooks/azure_t05_pageindex_run.ipynb`), then evaluated locally —
> see the **Run sequence** in the README.

## 8. KEY DESIGN NOTES

- **No vectors anywhere.** This method uses zero embeddings, zero FAISS, zero BM25. Retrieval is pure LLM reasoning.
- **Tree is deterministic.** Built from parsed metadata, not LLM-inferred. More reliable and reproducible than standard PageIndex.
- **LLM calls per query:** 3-4 minimum (law select + chapter select + article select + evaluate). Up to 3× that with iterations.
- **Context window budget:** Level 1 tree ~500-2000 tokens (depends on number of laws). Level 2 per law ~500-1000 tokens. Level 3 per chapter ~1000-3000 tokens. All well within LLaMA 3.1 8B's 128K context.
- **Temperature 0.0** for reproducibility. All LLM responses logged in reasoning_trace.
- **Cross-reference following** is PageIndex's key structural advantage over vector retrieval.
