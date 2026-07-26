#!/usr/bin/env bash
# One command to run the Arm-2C corrective-loop SMOKE on the compute instance:
# OLD navigate() vs NEW navigate_corrective() on a few queries (foreground — short).
#
#   bash corrective/launch_smoke.sh --stem 1867_06_08_1867060850 --qids 1048,202,240
#   bash corrective/launch_smoke.sh --stem 1804_03_21_1804032150 --qids 1,2,3 --max-rounds 2
#   bash corrective/launch_smoke.sh --stem 2003_07_17_2013A31614 --qids 1,2,3   # neg. control
#
# Resets + pins Ollama (num_ctx 16384, keep_alive=-1) so prompts aren't truncated
# (the old T05 4096 bug) and the model can't idle-unload, ensures the `ollama`
# package, pulls latest code, then runs the smoke and prints a per-query OLD-vs-NEW
# table. Needs Ollama + llama3.1:8b already installed on the instance.
#
# Mirrors launch_run.sh; nothing here is destructive (smoke writes only to
# runs/_corrective_smoke/).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../experiments/arm2c/corrective
ARM2C="$(dirname "$HERE")"
cd "$ARM2C" || exit 1

if [ "$#" -eq 0 ]; then
  echo "usage: bash corrective/launch_smoke.sh --stem <STEM> --qids <id,id,...> \\"
  echo "         [--mode enriched|bare] [--max-rounds 2] [--reseed-strategy ranked|all]"
  echo
  echo "stems: 1867_06_08_1867060850 (Penal)  1804_03_21_1804032150 (Civil)"
  echo "       1967_10_10_1967101055 (Jud-larger)  1967_10_10_1967101056 (Jud-smaller)"
  echo "       2003_07_17_2013A31614 (Housing / negative control)"
  exit 1
fi

echo "== reset Ollama + pin model @ num_ctx 16384 (clears any wedge, avoids 4096 truncation) =="
sudo systemctl restart ollama && sleep 8
curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","options":{"num_ctx":16384},"keep_alive":-1,"prompt":"ok","stream":false}' >/dev/null
ollama ps

echo "== update code + ensure ollama package =="
git pull -q 2>/dev/null || true
python -c "import ollama" 2>/dev/null || pip install -q ollama

echo "== run smoke: OLD navigate() vs NEW navigate_corrective() =="
python corrective/smoke_corrective.py "$@"
