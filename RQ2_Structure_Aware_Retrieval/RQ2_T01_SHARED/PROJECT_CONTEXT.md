# 01 — SHARED COMPONENTS CONTEXT
## Embeddings, LLM, Vector Stores, BM25

---

## 1. PURPOSE

Shared utilities used by all sub-projects. Each component is a thin wrapper providing a consistent interface. All model loading is lazy (loaded on first use) and cached (loaded once per process).

## 2. DIRECTORY STRUCTURE

```
shared/
├── __init__.py
├── embeddings.py       # Embedding model loading + encoding
├── llm.py              # LLaMA 3.1 8B via Ollama wrapper
├── faiss_store.py       # FAISS index management
├── bm25_store.py        # BM25 index management
└── utils.py             # Logging, timing, text utilities
```

## 3. COMPONENT SPECIFICATIONS

### 3.1 embeddings.py

```python
class EmbeddingModel:
    def __init__(self, model_name: str, max_tokens: int = 512,
                 fallback_model: str | None = None):
        """Load sentence-transformers model. Lazy initialization.
        fallback_model: alternate model (e.g. BGE-M3) for inputs over max_tokens."""
        
    def encode(self, texts: list[str], batch_size: int = 32, 
               show_progress: bool = True) -> np.ndarray:
        """Encode texts to normalized vectors. Returns (N, dim) array."""
        
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query. Handles instruction-prefix if model requires it.
        For mE5-instruct models, prepends 'query: ' automatically."""
        
    @property
    def dimension(self) -> int:
        """Embedding dimensionality."""

    @property
    def tokenizer(self):
        """Underlying HuggingFace fast tokenizer (supports return_offsets_mapping
        for chunk-boundary alignment). Used by the orchestrator and Arm 1."""
```

**Models:**
- Primary: `intfloat/multilingual-e5-large-instruct` (512 token limit, 1024-dim)
- Fallback: `BAAI/bge-m3` (1024 token limit) — used when enriched articles exceed 512 tokens

**Important:** mE5-instruct requires `query: ` prefix for queries and `passage: ` prefix for documents. The wrapper handles this automatically.

### 3.2 llm.py

```python
class LLMClient:
    def __init__(self, model: str = "llama3.1:8b", 
                 base_url: str = "http://localhost:11434",
                 temperature: float = 0.0,
                 num_ctx: int | None = 16384):
        """Ollama client wrapper. num_ctx=16384 overrides Ollama's 4096 default so
        long T05 PageIndex chapter-selection prompts (6-9k tokens) aren't truncated."""
        
    def generate(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        """Generate completion. Returns LLMResponse with text + usage stats."""
        
    def generate_json(self, prompt: str, system_prompt: str = None) -> dict:
        """Generate and parse JSON response. Handles markdown fences, retries on parse failure."""

@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
```

**Design decisions:**
- Temperature 0.0 always (reproducibility)
- JSON mode: strip ```json fences before parsing, retry once on parse failure
- All calls are logged with token counts for cost tracking

### 3.3 faiss_store.py

```python
class FAISSStore:
    def __init__(self, dimension: int):
        """Create empty FAISS IndexFlatIP store."""
    
    def add(self, ids: list[str], vectors: np.ndarray, metadata: list[dict]):
        """Add vectors with IDs and metadata. Maintains parallel metadata DataFrame."""
    
    def search(self, query_vector: np.ndarray, k: int = 100,
               filter_fn: callable = None, fetch_k: int = 500) -> list[SearchResult]:
        """Search with optional metadata filter.
        filter_fn takes metadata dict, returns bool.
        fetch_k: retrieve this many before filtering (FAISS does post-filtering)."""
    
    def save(self, path: str):
        """Save index + metadata to disk."""
    
    @classmethod
    def load(cls, path: str) -> 'FAISSStore':
        """Load index + metadata from disk."""

@dataclass
class SearchResult:
    id: str
    score: float
    metadata: dict
```

**Note:** FAISS with IndexFlatIP does post-filtering (retrieves fetch_k candidates, then filters by metadata). For the BSARD corpus size (~22K articles), this is efficient. Set fetch_k = 5× target k when filters are active.

### 3.4 bm25_store.py

```python
class BM25Store:
    def __init__(self, k1: float = 1.5, b: float = 0.25):
        """Create empty BM25 index. Defaults match RQ1 canonical hybrid."""
    
    def build(self, ids: list[str], texts: list[str], metadata: list[dict]):
        """Tokenize and index all documents."""
    
    def search(self, query: str, k: int = 100,
               filter_fn: callable = None) -> list[SearchResult]:
        """BM25 search with optional metadata filter."""
    
    def save(self, path: str):
        """Pickle index to disk (includes bm25_params for cache versioning)."""
    
    @classmethod
    def load(cls, path: str) -> 'BM25Store':
        """Load pickled index. Pre-CN-T01-001 caches load with a warning."""
```

**Hyperparameters (CN-T01-001, decided 2026-05-01):** `k1=1.5`, `b=0.25`. Matches RQ1's canonical hybrid. The rank_bm25 default `b=0.75` over-penalises long bodies in our corpus (22.7% of BSARD article bodies exceed 512 tokens; T04 review #8 showed Q38's GT slipping to rank 11 under the default). Existing pickles built before this change lack `bm25_params` and load with a warning — rebuild to pick up the new default.

**Tokenization (CN-T01-001, decided 2026-05-01):** Simple lowercased regex split + ~30-word French function-word stoplist. **Deliberately diverges from RQ1's spaCy `fr_core_news_lg` lemmatize + legal-keep stoplist.** Rationale: RQ1 parity would add spaCy + a 580 MB model as a runtime dep across every arm and force a full cache rebuild, with no empirical evidence that the simpler tokenizer is materially distorting RQ2 rankings (unlike `b=0.75`, which had a measured impact). RQ2 already documents three other deliberate divergences from RQ1 (mE5-instruct vs mE5-large, no `concat_2x` doc weighting, no neural reranking — the cross-encoder was excluded from the RQ2 architecture entirely); this is a fourth. Direct numeric comparison to RQ1's published BM25 numbers requires the "different tokenizer" caveat.

### 3.5 utils.py

```python
def first_sentence(text: str) -> str:
    """Extract first sentence from text. Handles abbreviations (Art., al., etc.)."""

def timer(func):
    """Decorator that logs execution time."""

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Configure logger with consistent format."""

def set_seeds(seed: int = 42):
    """Fix random seeds for numpy, torch, random."""
```
