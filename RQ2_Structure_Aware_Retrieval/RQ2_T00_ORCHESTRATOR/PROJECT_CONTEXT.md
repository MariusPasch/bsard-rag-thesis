# 00 — ORCHESTRATOR CONTEXT
## Pipeline Coordination, CLI, and Configuration

---

## 1. PURPOSE

The orchestrator is the main entry point for the RQ2 pipeline. It loads configuration, runs auto-discovery, dispatches work to sub-projects, and collects results for evaluation. It contains no retrieval logic itself.

## 2. CLI INTERFACE

```
Usage:
  python run_experiment.py                                    # all methods, all PDFs
  python run_experiment.py --methods arm1 2A 2B               # specific methods
  python run_experiment.py --pdf 2.pdf                        # specific PDF only
  python run_experiment.py --variants 2A-raw 2A-full 2A-terms # specific 2A variants
  python run_experiment.py --config custom_config.yaml        # custom config
  python run_experiment.py --skip-indexing                    # reuse existing indices
  python run_experiment.py --force-reindex                    # rebuild all indices even if present
  python run_experiment.py --eval-only                        # skip retrieval, run eval on saved results
```

(`--methods` choices are `arm1`, `2A`, `2B`. The module lives at
`src/orchestrator/run_experiment.py`; invoke it from the project root.)

## 3. EXECUTION FLOW

```python
def main(args):
    # 1. Load config
    config = load_config(args.config or "config.yaml")
    
    # 2. Auto-discover documents (calls 02_DATA_LOADER)
    document_bundles = data_loader.load_documents(config)
    # Returns list of DocumentBundle, each linking a PDF to its metadata + definitions
    
    # 3. For each document bundle:
    all_results = {}
    for bundle in document_bundles:
        if args.pdf and bundle.pdf_filename not in args.pdf:
            continue
        
        # 3a. Arm 1: Naive chunking (always runs if PDF available)
        if "arm1" in args.methods:
            results_arm1 = arm1_naive.run_arm1(bundle, config)
            all_results[f"{bundle.document_id}_arm1"] = results_arm1
        
        # 3b. Arm 2 methods (only if Azure extraction available)
        if bundle.has_azure_extraction:
            
            if "2A" in args.methods:
                for variant in args.variants or config['arm2']['metadata_filtering']['variants']:
                    results_2a = arm2_metadata.run_metadata_filtering(bundle, config, variant=variant)
                    all_results[f"{bundle.document_id}_2A_{variant}"] = results_2a
            
            if "2B" in args.methods:
                results_2b = arm2_pageindex.run_pageindex(bundle, config)
                all_results[f"{bundle.document_id}_2B"] = results_2b
        else:
            logger.warning(f"No Azure extraction for {bundle.pdf_filename}, skipping Arm 2")
    
    # 4. Evaluation (calls 07_EVALUATION)
    ground_truth = load_ground_truth(config) if ground_truth_exists(config) else None
    eval_report = evaluation.evaluate(all_results, ground_truth, config)
    
    # 5. Save results
    save_results(all_results, eval_report, config)
    
    # 6. Generate comparison tables and visualizations
    evaluation.generate_comparison(eval_report, config)
```

## 4. CONFIGURATION FILE (config.yaml)

```yaml
# Input paths
data:
  pdf_dir: "../RQ2_T02_DATA_LOADER/data/pdfs/"
  documents_csv: "../RQ2_T02_DATA_LOADER/data/csv/MyDocuments.csv"
  definitions_csv: "../RQ2_T02_DATA_LOADER/data/csv/DocumentDefinitions.csv"
  pdf_document_map: "../RQ2_T02_DATA_LOADER/data/csv/pdf_document_map.csv"
  ground_truth_dir: "../RQ2_T07_EVALUATION/ground_truth/"          # full split GT (scans *.json, skips schema.json)
  # ground_truth_file: "../RQ2_T07_EVALUATION/ground_truth/runs/<run_name>.json"  # optional curated subset; wins over ground_truth_dir
  bsard_db: "$RQ2_DATA_DIR/bsard_corpus.db"
  index_dir: "data/indices/"

# Models — used by 01_SHARED
models:
  embedding_model: "intfloat/multilingual-e5-large-instruct"
  embedding_model_fallback: "BAAI/bge-m3"
  embedding_max_tokens: 512
  llm_model: "llama3.1:8b"
  llm_temperature: 0.0
  ollama_base_url: "http://localhost:11434"

# Arm 1 — used by 03_ARM1_NAIVE
arm1:
  strategy: "sliding_window"      # "sliding_window" or "recursive"
  window_size: 512                # tokens (sliding_window only)
  stride: 256                     # tokens (sliding_window only)
  max_tokens: 512                 # tokens (recursive only)
  retrieval_top_k: 100
  top_k: 100

# Arm 2 — used by 04/05 sub-projects
arm2:
  metadata_filtering:
    variants: ["raw", "enriched", "filtered", "full", "terms"]
    boost_config:
      term_match: 1.3
      used_in: 1.5
      jurisdiction: 1.1
    retrieval_top_k: 100
    top_k: 100

  pageindex:
    max_iterations: 3
    max_laws_selected: 3
    max_chapters_selected: 5
    skip_law_selection_if_single_doc: true

# Evaluation — used by 07_EVALUATION
evaluation:
  metrics: ["recall@100", "recall@10", "mrr@10", "ndcg@10"]
  use_autonomous_eval: true
  stratified_analysis: true

# Output
output:
  results_dir: "data/results/"
  logs_dir: "data/logs/"
  save_traces: true
  save_retrieval_lists: true
```

## 5. PROJECT DIRECTORY STRUCTURE

Each sub-project is a **separate sibling package/repo**, installed editable into this
project's venv (see README Step 4). The orchestrator itself contains only:

```
RQ2_T00_ORCHESTRATOR/
├── config.yaml                  # Default configuration
├── requirements.txt
├── src/orchestrator/
│   ├── __init__.py
│   ├── run_experiment.py        # main entry point
│   ├── status.py                # read-only per-PDF status registry CLI
│   └── paths.py                 # RQ2 root + per-project data-dir helpers
└── data/                        # link into the shared RQ2 data root (gitignored)
    ├── indices/                 # Arm 1 index cache (config.data.index_dir)
    ├── results/                 # Output result JSONs (config.output.results_dir)
    ├── logs/                    # Run logs (config.output.logs_dir)
    └── status/                  # Per-PDF status registries from orchestrator.status
```

Sibling packages imported at runtime: `shared` (T01), `data_loader` (T02),
`arm1_naive` (T03), `arm2_metadata` (T04), `arm2_pageindex` (T05), `evaluation` (T07).
Source PDFs/CSVs are owned by T02; ground truth by T07. T06_ARM_RESULTS is a separate
post-pipeline consolidation step and is not imported here.

## 6. DEPENDENCIES (requirements.txt)

The orchestrator's own `requirements.txt` (the arms pull their heavier deps via their
own editable installs):

```
# Core
pandas>=2.0
numpy>=1.26
scipy>=1.11
pyyaml>=6.0
tqdm>=4.66

# PDF processing
PyMuPDF>=1.23

# Embeddings & retrieval
sentence-transformers>=2.7
faiss-cpu>=1.8
rank-bm25>=0.2
transformers>=4.40

# Agentic framework
langgraph>=0.1
langchain>=0.2
langchain-community>=0.2

# LLM backend (Ollama must be running separately)
ollama>=0.2

# Graph
networkx>=3.3
```
