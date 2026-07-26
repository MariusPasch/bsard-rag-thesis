#!/usr/bin/env bash
# Embedding-rerank vs LLM-rerank over the shipped reached pool (the re-SELECT lever).
# Needs Arm-2A's e5 encoder (sentence-transformers + the model), NOT Ollama — it
# reranks the pool reconstructed from the SAVED traces, so no navigation is re-run.
#
#   bash corrective/launch_embed_rerank.sh --stem 1804_03_21_1804032150
#   bash corrective/launch_embed_rerank.sh --stem 1804_03_21_1804032150 --text-field summary
#   bash corrective/launch_embed_rerank.sh --stem 1867_06_08_1867060850   # Penal
#
# First run downloads intfloat/multilingual-e5-large-instruct (~2 GB). SentenceTransformer
# auto-uses the GPU if torch sees CUDA. Runs over the full saved query set.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM2C="$(dirname "$HERE")"
cd "$ARM2C" || exit 1

if [ "$#" -eq 0 ]; then
  echo "usage: bash corrective/launch_embed_rerank.sh --stem <STEM> [--text-field text|summary] [--limit N]"
  echo "stems: 1804_..2150 Civil | 1867_..0850 Penal | 1967_..1055 Jud-larger |"
  echo "       1967_..1056 Jud-smaller | 2003_..1614 Housing"
  exit 1
fi

# Stop transformers from importing TensorFlow/Flax (that chain pulls NumPy-1.x
# binaries that crash under NumPy 2 on the azureml env). We only need the torch path.
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_ADVISORY_WARNINGS=1 TOKENIZERS_PARALLELISM=false

echo "== update code + ensure sentence-transformers =="
git pull -q 2>/dev/null || true
python -c "import sentence_transformers" 2>/dev/null || pip install -q sentence-transformers

# Installing sentence-transformers can upgrade NumPy to 2.x, which breaks the env's
# NumPy-1.x-built pyarrow/pandas/tensorflow. Pin back to <2 if that happened.
NPV="$(python -c 'import numpy,sys; sys.stdout.write(numpy.__version__.split(".")[0])' 2>/dev/null || echo 2)"
if [ "$NPV" -ge 2 ]; then
  echo "== numpy ${NPV}.x detected — pinning numpy<2 to match the env's compiled deps =="
  pip install -q "numpy<2"
fi

echo "== embed-rerank vs LLM-rerank over the shipped reached pool =="
python corrective/eval_embed_rerank.py "$@"
