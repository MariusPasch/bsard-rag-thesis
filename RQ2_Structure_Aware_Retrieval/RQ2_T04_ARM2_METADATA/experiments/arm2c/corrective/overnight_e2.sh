#!/usr/bin/env bash
# Overnight E2 for the 4 remaining PDFs: ALL corrective navigations first (Ollama
# only), THEN all e5 reranks (e5 only) — decoupled so the two models never share
# VRAM. Smallest doc first, so early PDFs finish even if the night is cut short.
# Fully resumable: e2_navigate skips queries already saved, so just re-run this
# whole script to resume. (Code Civil 1804 is already done — not included.)
#
#   nohup bash corrective/overnight_e2.sh > overnight_e2.log 2>&1 &
#   tail -f overnight_e2.log         # or monitor the per-stem nav.json counts
#
# REQUIRED for an overnight run: disable the compute instance's idle shutdown
# (Studio -> Compute -> instance -> Idle shutdown), else it stops mid-run and you
# must relaunch to resume.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM2C="$(dirname "$HERE")"
cd "$ARM2C" || exit 1

# smallest -> largest (65, 71, 133, 204 questions)
STEMS=(1867_06_08_1867060850 1967_10_10_1967101056 2003_07_17_2013A31614 1967_10_10_1967101055)
MAXROUNDS=2

echo "================ OVERNIGHT E2  $(date) ================"
git pull -q 2>/dev/null || true

# ---------- PHASE 1: corrective navigation (Ollama only) ----------
echo "== reset Ollama + pin model @ num_ctx 16384 =="
sudo systemctl restart ollama && sleep 8
curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","options":{"num_ctx":16384},"keep_alive":-1,"prompt":"ok","stream":false}' >/dev/null
python -c "import ollama" 2>/dev/null || pip install -q ollama
ollama ps

for stem in "${STEMS[@]}"; do
  echo "----------------------------------------------------------------"
  echo "== PHASE 1 navigate: $stem  $(date) =="
  python corrective/e2_navigate.py --stem "$stem" --all --max-rounds "$MAXROUNDS"
done

# ---------- PHASE 2: e5 rerank (e5 only) ----------
echo "================ PHASE 2: e5 rerank  $(date) ================"
export USE_TF=0 USE_FLAX=0 TRANSFORMERS_NO_ADVISORY_WARNINGS=1 TOKENIZERS_PARALLELISM=false
python -c "import sentence_transformers" 2>/dev/null || pip install -q sentence-transformers
NPV="$(python -c 'import numpy,sys; sys.stdout.write(numpy.__version__.split(".")[0])' 2>/dev/null || echo 2)"
if [ "$NPV" -ge 2 ]; then echo "== pinning numpy<2 =="; pip install -q "numpy<2"; fi

for stem in "${STEMS[@]}"; do
  echo "----------------------------------------------------------------"
  echo "== PHASE 2 rerank: $stem =="
  python corrective/e2_rerank.py --stem "$stem"
done

# ---------- summary ----------
echo "================ DONE  $(date) — per-PDF E2 summary ================"
for stem in 1804_03_21_1804032150 "${STEMS[@]}"; do
  md="runs/_corrective_e2/${stem}_e2_text.md"
  [ -f "$md" ] && { echo; echo "### $stem"; grep -E 'reach|R@10' "$md"; }
done
echo
echo "Back up the results:  git add -f runs/_corrective_e2/*_{nav.json,e2_text.*} && git commit -m 'E2 all PDFs' && git push"
