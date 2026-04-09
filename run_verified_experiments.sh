#!/bin/bash
# Run the full experimental matrix for a given model.
# Usage: bash run_verified_experiments.sh <MODEL_NAME> [--reasoning]

MODEL=$1
EXTRA_ARGS="${@:2}"

if [ -z "$MODEL" ]; then
    echo "Usage: bash run_verified_experiments.sh <MODEL_NAME> [--reasoning]"
    exit 1
fi

DATASETS=("./data/test.jsonl" "./data/test_hard.jsonl")
DATASET_NAMES=("original" "hard")
PIPELINES=("nl2regex" "nl2dsl2regex")
SHOTS=("zero" "five")
FEEDBACK_LEVELS=("binary" "counterexample" "diagnostic")

for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    DNAME=${DATASET_NAMES[$i]}

    if [ ! -f "$DATASET" ]; then
        echo "Skipping $DNAME: $DATASET not found"
        continue
    fi

    for PIPELINE in "${PIPELINES[@]}"; do
        for SHOT in "${SHOTS[@]}"; do
            # Baseline: single-shot, no verification, no constrained
            echo "=== $DNAME / $PIPELINE / $SHOT-shot / baseline ==="
            python scripts/run_inference_verified.py \
                --model-name "$MODEL" \
                --dataset-path "$DATASET" \
                --fewshot-path "./data/train.jsonl" \
                --pipeline "$PIPELINE" \
                --shot "$SHOT" \
                --feedback-level "binary" \
                --max-rounds 1 \
                $EXTRA_ARGS

            # Constrained only: single-shot
            echo "=== $DNAME / $PIPELINE / $SHOT-shot / constrained ==="
            python scripts/run_inference_verified.py \
                --model-name "$MODEL" \
                --dataset-path "$DATASET" \
                --fewshot-path "./data/train.jsonl" \
                --pipeline "$PIPELINE" \
                --shot "$SHOT" \
                --feedback-level "binary" \
                --max-rounds 1 \
                --constrained \
                $EXTRA_ARGS

            # Verification loop at each feedback level
            for FEEDBACK in "${FEEDBACK_LEVELS[@]}"; do
                echo "=== $DNAME / $PIPELINE / $SHOT-shot / $FEEDBACK / 5 rounds / constrained ==="
                python scripts/run_inference_verified.py \
                    --model-name "$MODEL" \
                    --dataset-path "$DATASET" \
                    --fewshot-path "./data/train.jsonl" \
                    --pipeline "$PIPELINE" \
                    --shot "$SHOT" \
                    --feedback-level "$FEEDBACK" \
                    --max-rounds 5 \
                    --constrained \
                    $EXTRA_ARGS
            done
        done
    done
done

echo "All experiments complete for $MODEL"
