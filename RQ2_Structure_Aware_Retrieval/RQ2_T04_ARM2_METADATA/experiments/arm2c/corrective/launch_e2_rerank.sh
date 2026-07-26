#!/usr/bin/env bash
# E2 phase 2 — e5 rerank over the saved corrective pools (e5 only, NO Ollama).
# Run AFTER launch_e2_nav.sh. Cheap: re-run for text-field / encoder sweeps.
#
#   bash corrective/launch_e2_rerank.sh --stem 1804_03_21_1804032150
#   bash corrective/launch_e2_rerank.sh --stem 1804_03_21_1804032150 --text-field summary

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM2C="$(dirname "$HERE")"
cd "$ARM2C" || exit 1
if [ "$#" -eq 0 ]; then
  echo "usage: bash corrective/launch_e2_rerank.sh --stem <STEM> [--text-field text|summary]"
  exit 1
fi

# torch-only path for transformers (avoid TF/NumPy-2 import breakage on azureml)
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_ADVISORY_WARNINGS=1 TOKENIZERS_PARALLELISM=false

echo "== update code + ensure sentence-transformers =="
git pull -q 2>/dev/null || true
python -c "import sentence_transformers" 2>/dev/null || pip install -q sentence-transformers
NPV="$(python -c 'import numpy,sys; sys.stdout.write(numpy.__version__.split(".")[0])' 2>/dev/null || echo 2)"
if [ "$NPV" -ge 2 ]; then echo "== pinning numpy<2 =="; pip install -q "numpy<2"; fi

echo "== E2 phase 2: e5 rerank over saved pools =="
python corrective/e2_rerank.py "$@"
