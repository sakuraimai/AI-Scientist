#!/usr/bin/env bash
# Pod GPU runner: sequential runs to keep the GPU busy overnight.
# Usage (on RunPod):
#   cd /workspace/AI-Scientist/templates/asian_ner
#   git pull
#   nohup bash run_pod.sh > pod_pipeline.log 2>&1 &

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/AI-Scientist}"
cd "${REPO_ROOT}/templates/asian_ner"

# RTX 3090 (24GB): 32 fits xlm-roberta-base NER at max_length=128.
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-1}"
NUM_SEEDS="${NUM_SEEDS:-1}"
MAX_EPOCHS="${MAX_EPOCHS:-2}"

run_one() {
  local out_dir="$1"
  local cap="$2"
  echo "===== $(date -Iseconds) START ${out_dir} (cap=${cap}) ====="
  python experiment.py \
    --out_dir "${out_dir}" \
    --seed "${SEED}" \
    --num_seeds "${NUM_SEEDS}" \
    --max_epochs "${MAX_EPOCHS}" \
    --max_train_per_lang "${cap}" \
    --batch_size "${BATCH_SIZE}" \
    2>&1 | tee "${out_dir}.log"
  echo "===== $(date -Iseconds) DONE ${out_dir} ====="
}

# run_1: hybrid clustering PoC (cap=500) + matched_random control
run_one run_1 500

# run_2: full WikiAnn (cap=0 = no cap)
run_one run_2 0

python plot.py
echo "===== $(date -Iseconds) Pipeline complete ====="
