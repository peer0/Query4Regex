#!/bin/bash

MODEL_NAME=$1

# 1) Pure LLM inference with normal instruction
python scripts/run_inference.py \
    --model-name "${MODEL_NAME}" \
    --dataset-path "./data/test.jsonl" \
    --pipeline "nl2dsl2regex" \
    --shot "zero"

# 2) Pure LLM inference with 5-shot
python scripts/run_inference.py \
    --model-name "${MODEL_NAME}" \
    --dataset-path "./data/test.jsonl" \
    --fewshot-path "./data/train.jsonl" \
    --pipeline "nl2dsl2regex" \
    --shot "five"
