#!/usr/bin/env bash
# Usage (on frappe, from ~/thesis/sae-uncertainty):
#   bash scripts/run_faith_gap.sh smoke   # 1 layer, 4 random draws (~minutes): checks the plumbing
#   bash scripts/run_faith_gap.sh full    # 6 layers x 40 random draws, split over GPU0/GPU1, background
set -u
cd "$(dirname "$0")/.."
source ~/venv-sae/bin/activate
mode=${1:-}
EXT=""
[ -f ../sae-halluc/results/intervene_L16_rag_matched.json ] && EXT="--external-json ../sae-halluc/results/intervene_L16_rag_matched.json"

if pgrep -u "$USER" -f "faith_gap.py|intervene_teacher_forced" >/dev/null; then
  echo "A faith_gap / intervention job is already running (GPU in use); refusing to start."; exit 1
fi
mkdir -p results

case "$mode" in
  smoke)
    python scripts/faith_gap.py --layers 16 --n-random 4 $EXT --output-dir results/faith_gap_smoke 2>&1 | tee results/faith_gap_smoke.log
    ;;
  full)
    CUDA_VISIBLE_DEVICES=0 nohup python scripts/faith_gap.py --layers 8,12,16 --n-random 40 $EXT \
        --output-dir results/faith_gap > results/faith_gap_gpu0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup python scripts/faith_gap.py --layers 20,24,28 --n-random 40 $EXT \
        --output-dir results/faith_gap > results/faith_gap_gpu1.log 2>&1 &
    echo "launched; logs: results/faith_gap_gpu{0,1}.log; summarize with:"
    echo "  python scripts/faith_gap.py --summarize-only --output-dir results/faith_gap --layers 8,12,16,20,24,28"
    ;;
  *) echo "usage: $0 smoke|full"; exit 2;;
esac
