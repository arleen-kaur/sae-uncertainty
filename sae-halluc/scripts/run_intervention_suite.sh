#!/usr/bin/env bash
# Runs the three L16 intervention variants (A: rag/matched null, B: rag/no boilerplate,
# C: no_rag) on two GPUs. Launch ONCE:   bash scripts/run_intervention_suite.sh
# Chain 1 (GPU0): A then C.   Chain 2 (GPU1): B.
set -u
cd "$(dirname "$0")/.."
source ~/venv-sae/bin/activate

FREQ=results/token_discovery_full_rag/token_features_predict.pkl
CSV=results/compare_rag_norag_predict.csv
for f in "$FREQ" "$CSV"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
done
if pgrep -u "$USER" -f intervene_teacher_forced >/dev/null; then
  echo "An intervention run is already running; refusing to start another."; exit 1
fi

COMMON=(--het-repo "$HOME/thesis/sae-uncertainty" --compare-csv "$CSV" --layer 16
        --split validation --max-responses 300 --n-random 10 --freq-pickle "$FREQ")
mkdir -p results

(
  CUDA_VISIBLE_DEVICES=0 python experiments/intervene_teacher_forced.py "${COMMON[@]}" \
      --condition rag --output results/intervene_L16_rag_matched.json > results/intervene_A.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python experiments/intervene_teacher_forced.py "${COMMON[@]}" \
      --condition no_rag --output results/intervene_L16_norag_matched.json > results/intervene_C.log 2>&1
) &

(
  CUDA_VISIBLE_DEVICES=1 python experiments/intervene_teacher_forced.py "${COMMON[@]}" \
      --condition rag --exclude-words based,passage,passages,provided,according,given \
      --output results/intervene_L16_rag_noboiler.json > results/intervene_B.log 2>&1
) &

echo "launched; logs: results/intervene_{A,B,C}.log"
wait
