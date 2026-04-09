#!/usr/bin/env python3
"""Generate the Query4Regex-Hard benchmark (4-7 ops)."""
import argparse
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query4regex.config import DEFAULT_ALPHABET
from query4regex.data.synth_hard import generate_hard_corpus
from query4regex.data.ambiguity_tagger import tag_ambiguity


def tag_corpus(path: str) -> None:
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    for rec in records:
        ops = rec.get("meta", {}).get("ops", [])
        instruction = rec.get("instruction", "")
        rec["meta"]["ambiguity_tags"] = tag_ambiguity(instruction, ops)

    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-dir", type=str, default="./data")
    args = parser.parse_args()

    test_path = os.path.join(args.out_dir, "test_hard.jsonl")
    train_path = os.path.join(args.out_dir, "train_hard.jsonl")

    print(f"Generating test set ({args.n_test} instances, 4-7 ops)...")
    test_stats = generate_hard_corpus(args.n_test, test_path, DEFAULT_ALPHABET, args.timeout)
    print(f"  Generated: {test_stats['total_generated']}, Filtered: {test_stats['total_filtered']}")
    print(f"  Per-op: {test_stats['per_op_generated']}")

    print(f"\nGenerating train set ({args.n_train} instances, 4-7 ops)...")
    train_stats = generate_hard_corpus(args.n_train, train_path, DEFAULT_ALPHABET, args.timeout)
    print(f"  Generated: {train_stats['total_generated']}, Filtered: {train_stats['total_filtered']}")

    print("\nTagging ambiguity types...")
    tag_corpus(test_path)
    tag_corpus(train_path)

    print(f"\nDone! Files: {test_path}, {train_path}")


if __name__ == "__main__":
    main()
