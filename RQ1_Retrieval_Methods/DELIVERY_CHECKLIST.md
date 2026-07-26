# Open-source delivery checklist

This document records how the RQ1 component was prepared for public release as part
of the BSARD RAG thesis mono-repo. The release follows four principles:

1. **Secrets.** No credentials in tracked files or git history; any API keys are read
   from environment variables, with a `.env.example` documenting the expected names.

2. **Portable data.** Large data artefacts (corpus DB, parquet exports, embeddings,
   FAISS indices, result JSONs) are not in git. They live in the companion Hugging Face
   dataset `mpaschalidis/bsard-rag-thesis-data` and download into a local gitignored data
   root. The RQ1 data root is the `BSARD_DATA_DIR` environment variable, defaulting to
   `<repo>/output`; `scripts/download_data.py` fetches the bundle into it. Source data is
   BSARD (CC BY-NC-SA 4.0).

3. **Open-source essentials.** Code is MIT-licensed with a data-license note; the README
   is written for clone → download → run on any platform; `requirements.txt` is pinned and
   the Python version is noted.

4. **Cross-component layout.** RQ1, RQ2 (`RQ2_Structure_Aware_Retrieval`), and RQ3
   (`RQ3_Autonomous_Evaluation`) are sibling components of the mono-repo. RQ1 installs the
   evaluation harness with `pip install -e "../RQ3_Autonomous_Evaluation"`.
