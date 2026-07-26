#!/usr/bin/env bash
# E2 — the full synthesis: Arm-2C corrective NAVIGATION (Ollama) + Arm-2A e5 RANKING.
# One corrective navigation per query yields two pools (single-pass + corrective);
# each is ranked by both the 8B and e5 → four R@10 numbers off one navigation.
#
#   bash corrective/launch_e2.sh --stem 1804_03_21_1804032150 --qids 243,158,1043,290
#   bash corrective/launch_e2.sh --stem 1867_06_08_1867060850 --qids 1048,202,240
#
# Needs BOTH Ollama+llama3.1:8b AND the e5 encoder (both fit a 16 GB T4). Foreground.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM2C="$(dirname "$HERE")"
cd "$ARM2C" || exit 1

if [ "$#" -eq 0 ]; then
  echo "usage: bash corrective/launch_e2.sh --stem <STEM> --qids <id,id,...> \\"
  echo "         [--mode enriched|bare] [--max-rounds 2] [--text-field text|summary]"
  exit 1
fi

# torch-only path for transformers (avoid the TF/NumPy-2 import breakage on azureml)
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_ADVISORY_WARNINGS=1 TOKENIZERS_PARALLELISM=false

echo "== reset Ollama + pin model @ num_ctx 16384 =="
sudo systemctl restart ollama && sleep 8
curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","options":{"num_ctx":16384},"keep_alive":-1,"prompt":"ok","stream":false}' >/dev/null
ollama ps

echo "== update code + ensure deps (ollama, sentence-transformers) =="
git pull -q 2>/dev/null || true
python -c "import ollama" 2>/dev/null || pip install -q ollama
python -c "import sentence_transformers" 2>/dev/null || pip install -q sentence-transformers
NPV="$(python -c 'import numpy,sys; sys.stdout.write(numpy.__version__.split(".")[0])' 2>/dev/null || echo 2)"
if [ "$NPV" -ge 2 ]; then echo "== pinning numpy<2 =="; pip install -q "numpy<2"; fi

echo "== E2: corrective nav (8B) + e5 rerank =="
python corrective/smoke_e2.py "$@"
