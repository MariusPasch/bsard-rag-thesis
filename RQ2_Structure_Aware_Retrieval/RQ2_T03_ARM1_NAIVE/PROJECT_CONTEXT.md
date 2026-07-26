# 03 — ARM 1 NAIVE CHUNKING CONTEXT
## PyMuPDF Extraction → Chunking → Best RQ1 Retrieval

---

## 1. PURPOSE

The baseline pipeline. Treats the PDF as flat text with no structural awareness. Represents the ceiling of what structure-agnostic retrieval can achieve. The key comparison: does Arm 2 (structure-aware) beat this?

## 2. DIRECTORY STRUCTURE

```
arm1_naive/
├── __init__.py
├── cache.py            # PDF-centric experiment cache: CacheRoot / PdfCache / ConfigCache,
│                       #   pdf_sha256 + config_hash, manifest validation, drift filter
├── chunker.py          # Chunking strategies + locate_articles() (BSARD article spans)
├── indexer.py          # FAISS + BM25 index construction for chunks (build/load_arm1_index)
└── retriever.py        # Cache-first hybrid retrieval pipeline (dense + sparse + RRF)
```

## 3. DEPENDENCIES

- Uses: `shared.embeddings`, `shared.faiss_store`, `shared.bm25_store`
- Input: `DocumentBundle.raw_text` (from 02_DATA_LOADER)
- Output: `list[RetrievalResult]`

## 4. CHUNKING (chunker.py)

### 4.1 Strategy A: Fixed-Size Sliding Window

```python
def chunk_sliding_window(text: str, tokenizer, 
                         window_size: int = 512, 
                         stride: int = 256) -> list[Chunk]:
    """
    Split text into overlapping windows of fixed token size.
    Returns Chunk objects with start/end token positions (needed for weighted metrics).
    """
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens) - window_size + 1, stride):
        chunk_tokens = tokens[i:i + window_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(Chunk(
            chunk_id=f"chunk_{len(chunks)}",
            text=chunk_text,
            source_pdf=...,
            start_token=i,
            end_token=i + len(chunk_tokens),
            start_char=...,  # compute from token-to-char mapping
            end_char=...,
        ))
    # Handle remainder
    if len(tokens) % stride != 0:
        # ... add final chunk
    return chunks
```

### 4.2 Strategy B: Recursive Text Splitting

```python
def chunk_recursive(text: str, max_tokens: int = 512) -> list[Chunk]:
    """
    Split by paragraph boundaries first, then by sentence if needed.
    Preserves natural text boundaries.
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if token_count(current_chunk + para) <= max_tokens:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(make_chunk(current_chunk))
            if token_count(para) > max_tokens:
                # Split long paragraph by sentences
                for sentence_group in split_sentences(para, max_tokens):
                    chunks.append(make_chunk(sentence_group))
                current_chunk = ""
            else:
                current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(make_chunk(current_chunk))
    return chunks
```

**Important:** Both strategies must record `start_token` and `end_token` positions for each chunk. These are required by the weighted metrics in 07_EVALUATION to compute chunk-article overlap weights.

## 5. INDEXING (indexer.py)

```python
def build_arm1_index(chunks: list[Chunk], embedding_model, config) -> tuple[FAISSStore, BM25Store]:
    """
    Build FAISS and BM25 indices over chunks.
    Returns both stores for hybrid retrieval.
    """
    # Embed all chunks
    texts = [chunk.text for chunk in chunks]
    vectors = embedding_model.encode(texts)
    
    # Build FAISS
    faiss_store = FAISSStore(dimension=embedding_model.dimension)
    faiss_store.add(
        ids=[c.chunk_id for c in chunks],
        vectors=vectors,
        metadata=[{"source_pdf": c.source_pdf, "start_token": c.start_token, 
                   "end_token": c.end_token} for c in chunks]
    )
    
    # Build BM25
    bm25_store = BM25Store()
    bm25_store.build(
        ids=[c.chunk_id for c in chunks],
        texts=texts,
        metadata=[{"source_pdf": c.source_pdf} for c in chunks]
    )
    
    return faiss_store, bm25_store
```

## 6. RETRIEVAL (retriever.py)

```python
def retrieve_arm1(query: str, faiss_store, bm25_store,
                  embedding_model, config) -> list[dict]:
    """
    Hybrid retrieval pipeline:
    1. Dense retrieval (FAISS top-k)
    2. Sparse retrieval (BM25 top-k)
    3. Reciprocal Rank Fusion
    """
    retrieval_top_k = config['arm1']['retrieval_top_k']  # 100
    top_k = config['arm1']['top_k']                       # 100
    
    # Dense retrieval
    query_vec = embedding_model.encode_query(query)
    dense_results = faiss_store.search(query_vec, k=retrieval_top_k)
    
    # Sparse retrieval
    sparse_results = bm25_store.search(query, k=retrieval_top_k)
    
    # RRF fusion → take top_k
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    return fused[:top_k]

def reciprocal_rank_fusion(results_a, results_b, k=60) -> list[SearchResult]:
    """
    Combine two ranked lists using RRF: score = Σ 1/(k + rank).
    Parameter-free fusion method.
    """
    scores = defaultdict(float)
    for rank, result in enumerate(results_a):
        scores[result.id] += 1.0 / (k + rank + 1)
    for rank, result in enumerate(results_b):
        scores[result.id] += 1.0 / (k + rank + 1)
    
    # Merge metadata from both sources
    all_results = {r.id: r for r in results_a + results_b}
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return [SearchResult(id=id, score=score, metadata=all_results[id].metadata) 
            for id, score in fused]
```

## 7. INTERFACE FOR ORCHESTRATOR

The actual entry point is **cache-aware** — it loads pre-built FAISS+BM25 from
`configs/<config_hash>/` when the manifest validates and skips extract/locate/chunk/embed
entirely:

```python
def run_arm1(
    bundle: DocumentBundle,
    embedding_model: EmbeddingModel,
    tokenizer,                       # HF fast tokenizer matching the embedding model
    db_path: Path,                   # bsard_corpus.db (for article -> bsard_id)
    index_dir: Path,                 # cache root (the data/ junction target)
    config: dict,
    force_reindex: bool = False,
    *,
    run_label: str | None = None,    # defaults to a UTC timestamp
) -> list[RetrievalResult]:
    """
    1. Compute pdf_sha256 + config_hash up front.
    2. If configs/<config_hash>/ validates -> load FAISS+BM25, skip the pipeline (hot path).
    3. Else run: extract -> locate_articles -> chunk -> save chunks -> build indices -> manifest.
    4. Retrieve per question; write RetrievalResult list to
       data/<doc_id>/results/<config_hash>/<run_label>.jsonl.
    """
```

(The orchestrator calls this with `force_reindex=args.force_reindex`. There are no LLM
calls, so every `RetrievalResult.cost` is zero.)

## 8. IMPORTANT NOTES

- **No LLM calls in Arm 1.** This is a pure embedding + BM25 + RRF pipeline. LLM cost = 0.
- **Chunk objects must track token positions** (`start_token`, `end_token`) for the weighted evaluation metrics in 07_EVALUATION.
- **The same embedding model** must be used as in Arm 2 for fair comparison.
- **RRF parameter k=60** is the standard default. Not tuned per experiment.

## 9. EVALUATION CAVEAT — DOC 2 GROUND-TRUTH DRIFT

**Do not misdiagnose near-zero recall on `2004_05_27_2004A27101.pdf` (doc_id 2) as a chunker or retriever issue.**

For the canonical sample document (Walloon Code de l'Environnement, **Livre 1er**), the BSARD ground truth for bsard_ids `D202 / D230 / D233 / D237 / D241 / D242 / D244 / D245` references content from a *different Livre* of the same code. Article numbers are reused across Livres, and the upstream BSARD extraction matches by article number alone — not by canonical text. Concretely, the PDF's `Art. D233` is about SPAQUE governance while BSARD's canonical `D233` is about water billing.

T03's `locate_articles` correctly anchors article spans by `_ART_RE` matches; the chunker correctly carves them up. But the *content* inside those spans is not what BSARD's GT references, so binary recall on the doc-2 question subset (Q35 / Q37 / Q38 / Q39) floors near zero by construction, regardless of chunking strategy or retriever quality. The cosine-on-4-grams analysis in T04 found the GT for our 4-question subset all sits in the 0.12–0.43 partial/drift band. The BSARD `question_extraction_status` JSONL independently marks all 5 doc-2-linked questions as `not_present`, corroborating the finding.

**Implication for benchmarking T03:** absolute recall numbers on doc 2 are not informative about T03's retrieval quality. To benchmark T03's actual ceiling, add a PDF to the AzureDI dump whose article numbers are unique within the BSARD corpus, or restrict the comparison to questions whose GT canonical text is a substantive match for content present in the indexed PDFs.

Full diagnosis: see the T04 drift analysis under `RQ2_T04_ARM2_METADATA/analysis/`
(cosine-on-4-grams / orphan-paragraph linker analysis).
