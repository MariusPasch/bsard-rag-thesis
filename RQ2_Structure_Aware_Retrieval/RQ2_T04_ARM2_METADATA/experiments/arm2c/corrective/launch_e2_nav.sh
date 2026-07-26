#!/usr/bin/env bash
# E2 phase 1 — corrective NAVIGATION only (Ollama, NO e5). Saves the reached pools
# so phase 2 (launch_e2_rerank.sh) can e5-rerank them without VRAM contention.
#
#   bash corrective/launch_e2_nav.sh --stem 1804_03_21_1804032150 \
#        --qids 243,244,252,181,290,1057,158,159,1043,302 --max-rounds 2
#   bash corrective/launch_e2_nav.sh --stem 1804_03_21_1804032150 --all   # full set (resumable)
#
# Resumable: re-running skips queries already saved. Needs Ollama+llama3.1:8b only.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM2C="$(dirname "$HERE")"
cd "$ARM2C" || exit 1
if [ "$#" -eq 0 ]; then
  echo "usage: bash corrective/launch_e2_nav.sh --stem <STEM> (--qids <id,...> | --all) [--max-rounds 2]"
  exit 1
fi

echo "== reset Ollama + pin model @ num_ctx 16384 =="
sudo systemctl restart ollama && sleep 8
curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","options":{"num_ctx":16384},"keep_alive":-1,"prompt":"ok","stream":false}' >/dev/null
ollama ps

echo "== update code + ensure ollama package =="
git pull -q 2>/dev/null || true
python -c "import ollama" 2>/dev/null || pip install -q ollama

echo "== E2 phase 1: corrective navigation (save pools) =="
python corrective/e2_navigate.py "$@"
