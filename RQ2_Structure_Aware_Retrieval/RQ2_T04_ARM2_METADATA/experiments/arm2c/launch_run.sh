#!/usr/bin/env bash
# One command to (re)start the detached Arm-2C run on the compute instance.
# Re-runnable: resets Ollama, pins the model, and resumes any q*.json on disk.
#
#   bash launch_run.sh                      # enriched + rerank (default)
#   bash launch_run.sh --mode bare          # custom args passed to run_all.py
#   bash launch_run.sh --mode enriched      # enriched, no rerank
#
# Survives terminal close / disconnect / laptop sleep (nohup). Only an instance
# shutdown stops it — then just run this again, it resumes.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
ARGS="${*:---mode enriched --rerank}"

echo "== reset Ollama + pin model @ num_ctx 16384 (clears any wedge) =="
sudo systemctl restart ollama && sleep 8
curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.1:8b","options":{"num_ctx":16384},"keep_alive":-1,"prompt":"ok","stream":false}' >/dev/null
ollama ps

echo "== update code + ensure ollama package =="
git pull -q 2>/dev/null
python -c "import ollama" 2>/dev/null || pip install -q ollama

# refuse to double-launch
if pgrep -f "run_all.py" >/dev/null; then
  echo "!! run_all.py is ALREADY running — not launching again. Monitor instead:"
  echo "   tail -f $(pwd)/run.log"
  exit 0
fi

echo "== launch detached: run_all.py $ARGS =="
nohup python run_all.py $ARGS > run.log 2>&1 &
sleep 4
echo "---- first log lines ----"
tail -8 run.log
echo "-------------------------"
echo "watch : tail -f $(pwd)/run.log"
echo "count : ls /home/azureuser/results/*/q*.json | wc -l"
