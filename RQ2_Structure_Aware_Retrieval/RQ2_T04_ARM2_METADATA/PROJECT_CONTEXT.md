# 04 — ARM 2A METADATA-FILTERED RETRIEVAL CONTEXT
## Enriched Article Embeddings + Query-Time Filtering

---

## 1. PURPOSE

Article-level retrieval with two layers of metadata exploitation: enriching article embeddings at index time (prepending document context and term definitions) and filtering/boosting at query time (term dictionary matching, regex extraction, LLM classification). Runs in multiple variants to ablate each contribution.

## 2. DIRECTORY STRUCTURE

```
arm2_metadata/
├── __init__.py
├── azuredi_loader.py    # Load AzureDI dump (MyDocuments/Definitions/VectorDB_*.json) → AzureNode list
├── bsard_link.py        # AzureNode → bsard_id resolver (article-number lookup vs BSARD DB)
├── enricher.py          # Metadata-prepend variants for embedding input (enrich_unit)
├── query_extractor.py   # Term matching + regex + LLM classifier (extract_query_signals)
├── boost.py             # Soft boost / hard filter application (apply_filters_and_boosts)
├── indexer.py           # FAISS + BM25 over node/article units (build_arm2_index)
├── cache.py             # Doc / config caches (mirrors T03 layout under data/<doc_id>/)
└── retriever.py         # run_arm2_metadata: full pipeline + persistence
```

## 3. DEPENDENCIES

- Uses: `shared.embeddings`, `shared.faiss_store`, `shared.bm25_store`, `shared.llm`; T03's
  `compute_config_hash` for cache keys
- Input: the **AzureDI dump** loaded via `azuredi_loader.load_azuredi_corpus` → `AzureNode`
  units (indexed as `node` or aggregated to `article`). It does **not** consume T02's
  `DocumentBundle.articles` — T04 reads the AzureDI metadata directly.
- Output: `list[RetrievalResult]`

## 4. INDEX-TIME ENRICHMENT (enricher.py)

**Six** embedding-input variants — `raw`, `enriched`, `summary`, `filtered`, `full`,
`terms` — produced by `enrich_unit(...)` over an `AzureNode` **or** `Article` unit (the
`summary` variant is node-only; it consumes the AzureDI `content_summary` + `keywords`
fields). The simplified per-variant logic below still applies; the real entry point is
`enrich_unit`, with a token-budget truncation cascade (`_truncate_to_budget`) and
`EnrichmentStats` telemetry:

```python
def enrich_unit(unit: AzureNode | Article, variant: str, ...) -> str:
    """Generate embedding input text for a given variant (illustrative logic below)."""
    
    if variant == "raw":
        return article.text
    
    elif variant == "enriched":
        return (
            f"[Document: {article.document_title}]\n"
            f"[Authority: {article.issuing_authority}]\n"
            f"[Jurisdiction: {article.jurisdiction}]\n"
            f"[Reference: {article.document_number}]\n"
            f"[In force: {article.entry_into_force}]\n\n"
            f"{article.text}"
        )
    
    elif variant == "filtered":
        # Same as raw — filtering happens at query time only
        return article.text
    
    elif variant == "full":
        # Same as enriched — filtering also happens at query time
        return (
            f"[Document: {article.document_title}]\n"
            f"[Authority: {article.issuing_authority}]\n"
            f"[Jurisdiction: {article.jurisdiction}]\n"
            f"[Reference: {article.document_number}]\n\n"
            f"{article.text}"
        )
    
    elif variant == "terms":
        terms_block = "\n".join(
            f"- {term}: {defn}"
            for term, defn in article.term_definitions.items()
        ) if article.term_definitions else ""
        
        header = (
            f"[Document: {article.document_title}]\n"
            f"[Authority: {article.issuing_authority}]\n"
            f"[Jurisdiction: {article.jurisdiction}]\n"
            f"[Reference: {article.document_number}]\n"
        )
        
        if terms_block:
            return f"{header}\n{article.text}\n\n[Defined terms in this article]\n{terms_block}"
        else:
            return f"{header}\n{article.text}"
```

**Token length handling:** If enriched input exceeds the embedding model's max_tokens (512 for mE5), truncate in this order: (1) terms block first, (2) then article text from the end. Log truncation events. If truncation is frequent, switch to BGE-M3 (1024 tokens).

## 5. QUERY-TIME EXTRACTION (query_extractor.py)

Three extraction methods, applied in sequence:

```python
@dataclass
class QuerySignals:
    matched_terms: list[str] = field(default_factory=list)
    term_article_ids: set[str] = field(default_factory=set)
    jurisdiction: str | None = None
    document_number: str | None = None
    date_ref: str | None = None
    legal_domain: str | None = None

def extract_query_signals(query: str, definitions_df: pd.DataFrame,
                          llm_client=None) -> QuerySignals:
    signals = QuerySignals()
    
    # Method 1: Term dictionary matching (deterministic, zero-cost)
    signals = _extract_term_signals(query, definitions_df, signals)
    
    # Method 2: Rule-based regex
    signals = _extract_regex_signals(query, signals)
    
    # Method 3: LLM classification (only if methods 1+2 found nothing)
    if llm_client and not signals.has_any_filter():
        signals = _extract_llm_signals(query, llm_client, signals)
    
    return signals
```

### Method 1: Term Dictionary Matching

```python
def _extract_term_signals(query: str, definitions_df: pd.DataFrame,
                          signals: QuerySignals) -> QuerySignals:
    query_lower = query.lower()
    for _, row in definitions_df.iterrows():
        term_match = row['Term'].lower() in query_lower
        translated_match = row['TranslatedTerm'].lower() in query_lower
        if term_match or translated_match:
            signals.matched_terms.append(row['Term'])
            used_in = json.loads(row['UsedIn'])
            signals.term_article_ids.update(str(uid) for uid in used_in)
    return signals
```

### Method 2: Rule-Based Regex

```python
PATTERNS = {
    "document_number": [
        r"(?:loi|décret|arrêté|decision|règlement)\s+(?:du\s+)?\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
        r"(?:DECISION|REGULATION|DIRECTIVE)\s*\(?(?:EU|EC|EEC)\)?\s*(?:No\.?\s*)?\d{4}[/-]\d+",
    ],
    "article_ref": [
        r"(?:art(?:icle)?\.?\s*)[D\.]?\s*\d+",
    ],
    "date_ref": [
        r"\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}",
    ],
    "jurisdiction_signal": [
        r"(?:wallon(?:ne)?|flamand(?:e)?|bruxell(?:ois|es)?|fédéral(?:e)?|européen(?:ne)?|belge)",
    ],
}
```

### Method 3: LLM Classification

```python
LLM_CLASSIFY_PROMPT = """Given this legal question: "{query}"

Extract the following if identifiable:
- jurisdiction: EU / Belgian Federal / Walloon / Flemish / Brussels / Unknown
- legal_domain: environmental / civil / criminal / commercial / administrative / Unknown
- temporal_scope: specific date or "current"

Respond in JSON only, no other text."""
```

## 6. FILTERING AND BOOSTING (boost.py)

```python
def apply_filters_and_boosts(
    results: list[SearchResult],
    signals: QuerySignals,
    boost_config: dict
) -> list[SearchResult]:
    """
    Apply hard filters and soft boosts to retrieval results.
    """
    # Hard filters (remove non-matching)
    filtered = results
    if signals.jurisdiction:
        # Only hard-filter if confidence is high (explicit mention)
        filtered = [r for r in filtered 
                    if r.metadata.get('jurisdiction', '').lower() == signals.jurisdiction.lower()
                    or not r.metadata.get('jurisdiction')]  # keep if no metadata
    
    # Always filter out non-effective documents
    filtered = [r for r in filtered 
                if r.metadata.get('regulatory_status', 'Effective') == 'Effective']
    
    # Soft boosts (multiply scores)
    for result in filtered:
        meta_terms = set(result.metadata.get('defined_terms', []))
        query_terms = set(signals.matched_terms)
        
        # Term match boost
        if meta_terms & query_terms:
            result.score *= boost_config.get('term_match', 1.3)
        
        # UsedIn set membership boost
        if result.id in signals.term_article_ids:
            result.score *= boost_config.get('used_in', 1.5)
        
        # Jurisdiction match boost (soft, from LLM classifier)
        if signals.jurisdiction and result.metadata.get('jurisdiction') == signals.jurisdiction:
            result.score *= boost_config.get('jurisdiction', 1.1)
    
    return sorted(filtered, key=lambda r: r.score, reverse=True)
```

## 7. EXPERIMENTAL VARIANTS

| Variant | Embedding | Query Filter | What It Tests |
|---|---|---|---|
| `raw` | Raw body | None | Node/article-boundary benefit only |
| `enriched` | Doc-context header + body | None | Index-time enrichment alone |
| `summary` | Header + AzureDI summary + keywords + body (node-only) | None | Free AzureDI metadata |
| `filtered` | Raw body | Full filter pipeline | Query-time filtering alone |
| `full` | Doc-context header + body | Full filter pipeline | Combined enrichment + filtering |
| `terms` | Header + body + defined-terms block | Full filter pipeline | Full pipeline with term definitions |

(Each variant runs on both `unit=node` and `unit=article`, except `summary` which is node-only.)

## 8. INTERFACE FOR ORCHESTRATOR

The real entry point is `retriever.run_arm2_metadata` — keyword-only, cache-aware per
`(doc_id, variant, unit)`, loading the AzureDI corpus directly:

```python
def run_arm2_metadata(
    *,
    azuredi_dir: Path, pdf_document_map_csv: Path, bsard_db_path: Path,
    pdf_path: Path | None,
    embedding_model: EmbeddingModel, tokenizer, llm,        # llm optional (query method 3)
    index_root: Path, config: dict,
    doc_id: str, azure_doc_id: int,
    variant: str, unit: str,                                # unit: "node" | "article"
    run_label: str | None = None,
    question_ids: set[str] | None = None, split: str | None = "test",
    force_reindex: bool = False, apply_boost_override: bool | None = None,
) -> list[RetrievalResult]:
    """Load AzureDI corpus -> enrich (enrich_unit) -> build_arm2_index -> retrieve_arm2
    (with extract_query_signals + apply_filters_and_boosts when the variant filters)
    -> persist results/<config_hash>/<run_label>.jsonl."""
```

> **Note:** the T00 orchestrator still imports a legacy `run_metadata_filtering(bundle,
> config, variant)` name that predates this API; in practice T04 is run standalone via its
> own scripts/notebook, not through `run_experiment.py`.

## 9. RESULTS

Evaluation: the **v4 5-stem sweep** (`LINKER_VERSION = 4`) across the curated
PDF set `{1967101056, 1867060850, 1804032150, 2013A31614, 1967101055}`
(doc_ids 5/6/9/8/7). Doc 2 (`2004A27101.pdf`) is excluded for GT-drift reasons
documented in the comparison writeup below (its BSARD ground truth references a
different Livre of the same code, flooring every retriever).

Headline: T04 (`arm2_metadata_node_summary` best overall) beats T03 on R@10
on 4/5 stems and on MRR@10 + nDCG@10 on 5/5; T03 retains R@100 on 4/5 stems.
Doc 6 (Code Pénal) is the lone stem where T03 wins R@10. See the writeups
below for the full picture.

| Artefact | What it has |
|---|---|
| [`analysis/comparison_T03_vs_T04_v4.md`](analysis/comparison_T03_vs_T04_v4.md) | Thesis-ready 5-stem aggregate tables + ≤200 word interpretation + caveats. |
| [`analysis/comparison_summary_v4.md`](analysis/comparison_summary_v4.md) | Headline findings (hand-written) + auto-generated per-metric tables, wins matrix, T04-vs-T03 head-to-head, best-variant ranking. |
| [`analysis/comparison_summary_v4.csv`](analysis/comparison_summary_v4.csv) | Machine-readable wide form of the above. |
| [`analysis/comparison_all_stems_v4.csv`](analysis/comparison_all_stems_v4.csv) | Long form: one row per (stem, method, metric, value). |
| [`data/comparison_t03_vs_t04_<stem>.csv`](data/) | Source per-stem CSVs (T03 + 6 T04 SMOKE_PLAN variants), one per curated stem. |
| [`data/comparison_per_query_<stem>.json`](data/) | Per-query GT + per-method metrics + T07 cosine GT (where populated). |
| [`scripts/aggregate_comparison.py`](scripts/aggregate_comparison.py) | Regenerates the three `comparison_*_v4.{csv,md}` artefacts from the per-stem CSVs. Preserves any hand-written preamble above the `AUTO-BEGIN` sentinel in the markdown. |

Deep-dive add-ons:

- [`analysis/error_analysis_doc6_v4.md`](analysis/error_analysis_doc6_v4.md) — per-question failure analysis for the doc 6 R@10 deficit. Finding: 8 of 9 T03-wins are multi-GT queries where T04 finds the first GT but adjacent sibling GTs rank 11–100 (`R@100 = 1.0` on all 8); 1 is a true coverage gap (Q718).
- [`analysis/v3_vs_v4_doc8_retro.md`](analysis/v3_vs_v4_doc8_retro.md) — side-by-side of doc 8 metrics under linker v3 vs v4 on the 75-question common subset. R@10 swings +0.55–0.65 absolute across all 6 variants; MRR@10 +0.52–0.61. Empirical confirmation of the `_ART_RE_AZURE` fix's impact.
- [`analysis/boost_ablation_v4.md`](analysis/boost_ablation_v4.md) — 5 stems × {raw, enriched, summary, full} × {boost off, on} sparse-only grid. Finding: boost stage is a structural no-op under default (regex-only, no LLM) signal extraction — Δ = 0 on every cell across all 5 stems and all 4 metrics. Index-time enrichment carries the full T04 signal; the boost stage is wired in but inert on BSARD-style informal questions.

