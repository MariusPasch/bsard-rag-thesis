#!/usr/bin/env bash
# Run enriched+rerank on several PDFs sequentially, detached + resume-friendly.
# Defaults to the 4 PDFs other than 1804 (already done).
#
#   nohup bash scale_run.sh > scale.log 2>&1 &      # the 4 remaining PDFs
#   nohup bash scale_run.sh 1867_06_08_1867060850 > scale.log 2>&1 &   # custom list
#   tail -f scale.log
#
# One Ollama reset/pin up front; the client's keep_alive/timeout/retry handle the rest.
# Each PDF resumes from its own q*.json, so a stop+relaunch continues where it left off.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
DOCS="${*:-1867_06_08_1867060850 1967_10_10_1967101055 1967_10_10_1967101056 2003_07_17_2013A31614}"

if pgrep -f "run_all.py" >/dev/null; then
  echo "!! run_all.py already running — not starting. Monitor: tail -f scale.log / run.log"
  exit 0
fi

echo "== reset Ollama + pin model @ num_ctx 16384 =="
sudo systemctl restart ollama && sleep 8
curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","options":{"num_ctx":16384},"keep_alive":-1,"prompt":"ok","stream":false}' >/dev/null
ollama ps

git pull -q 2>/dev/null
python -c "import ollama" 2>/dev/null || pip install -q ollama

for d in $DOCS; do
  echo "===================================================================="
  echo "===== $d  ($(date '+%H:%M:%S')) ====="
  echo "===================================================================="
  python run_all.py --doc-id "$d" --mode enriched --rerank
done
echo "===================================================================="
echo "===== ALL DONE ($(date '+%H:%M:%S')) ====="
echo "===================================================================="
