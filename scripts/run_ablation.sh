#!/usr/bin/env bash
# Architectural ablation runner: train + eval each of the three head
# variants on the 5-dataset zero-shot subset, aggregate into one CSV.
#
# Required: training data staged + GPU available.
#
# Override env vars:
#   TRAINING_DATA  comma-separated paths to .arrow files (TSMixup, KernelSynth)
#   SEED           random seed (default 0)
#   OUTPUT_BASE    where to write checkpoints + per-variant CSVs (default ./output/ablation)
#   RESULTS_DIR    where to write the aggregated CSV (default ./results/ablation)
#   MAX_STEPS      override training steps (default 25000 from the yaml)
#
# Example:
#   TRAINING_DATA=/data/tsmixup-data.arrow,/data/kernelsynth-data.arrow \
#       bash scripts/run_ablation.sh
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${TRAINING_DATA:?set TRAINING_DATA to comma-separated paths to TSMixup + KernelSynth arrow files}"
: "${SEED:=0}"
: "${OUTPUT_BASE:=./output/ablation}"
: "${RESULTS_DIR:=./results/ablation}"
: "${TRAIN_CONFIG:=scripts/training/configs/chronos-t5-tiny-ablation.yaml}"
: "${EVAL_CONFIG:=scripts/evaluation/configs/zero-shot-ablation.yaml}"

mkdir -p "$OUTPUT_BASE" "$RESULTS_DIR"

VARIANTS=(lm_head hidden mixture)

for variant in "${VARIANTS[@]}"; do
    out="$OUTPUT_BASE/$variant-seed$SEED"
    if [[ -d "$out/checkpoint-final" ]]; then
        echo "==> [$variant] checkpoint-final exists, skipping training"
    else
        echo "==> [$variant] training ($TRAIN_CONFIG, seed=$SEED)"
        python scripts/training/train.py \
            --config "$TRAIN_CONFIG" \
            --training-data-paths "$TRAINING_DATA" \
            --head-variant "$variant" \
            --output-dir "$out" \
            --seed "$SEED" \
            ${MAX_STEPS:+--max-steps "$MAX_STEPS"}
    fi
done

for variant in "${VARIANTS[@]}"; do
    out="$OUTPUT_BASE/$variant-seed$SEED"
    csv="$RESULTS_DIR/zero-shot-ablation-$variant-seed$SEED.csv"
    echo "==> [$variant] evaluating -> $csv"
    python scripts/evaluation/evaluate.py \
        "$EVAL_CONFIG" \
        "$csv" \
        --chronos-model-id "$out/checkpoint-final" \
        --head-variant "$variant" \
        --device cuda \
        --torch-dtype bfloat16 \
        --batch-size 32
done

echo "==> aggregating"
python scripts/aggregate_ablation.py \
    --results-dir "$RESULTS_DIR" \
    --pattern "zero-shot-ablation-*-seed$SEED.csv" \
    --out "$RESULTS_DIR/aggregate-seed$SEED.csv"

echo
echo "Done. Per-variant CSVs and aggregate-seed$SEED.csv in $RESULTS_DIR/"
