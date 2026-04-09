#!/usr/bin/env python3
"""
Analyze verification results by ambiguity type.

Usage:
    python scripts/analyze_ambiguity.py \
        --nl-results result/verified/nl2regex/five-shot/counterexample_r5/model.jsonl \
        --dsl-results result/verified/nl2dsl2regex/five-shot/counterexample_r5/model.jsonl \
        --dataset-path data/test.jsonl \
        --output result/analysis/
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query4regex.data.ambiguity_tagger import tag_ambiguity


def load_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_accuracy_at_k(results: List[Dict], max_k: int) -> Dict[int, float]:
    acc = {}
    for k in range(1, max_k + 1):
        correct = sum(
            1 for r in results
            if r.get("success") and r.get("solved_at_round", max_k + 1) <= k
        )
        acc[k] = correct / max(1, len(results))
    return acc


def analyze(nl_results, dsl_results, dataset, max_rounds):
    ambiguity_tags = {}
    for i, item in enumerate(dataset):
        ops = item.get("meta", {}).get("ops", [])
        tags = tag_ambiguity(item.get("instruction", ""), ops)
        ambiguity_tags[i] = tags

    all_types = ["scope", "anaphoric", "operator_scope", "implicit_ordering", "fragment"]

    analysis = {
        "overall": {
            "nl_accuracy_at_k": compute_accuracy_at_k(nl_results, max_rounds),
            "dsl_accuracy_at_k": compute_accuracy_at_k(dsl_results, max_rounds),
        },
        "per_type": {},
        "gap_at_k": {},
    }

    for k in range(1, max_rounds + 1):
        nl_acc = analysis["overall"]["nl_accuracy_at_k"][k]
        dsl_acc = analysis["overall"]["dsl_accuracy_at_k"][k]
        analysis["gap_at_k"][k] = dsl_acc - nl_acc

    for amb_type in all_types:
        type_indices = {i for i, tags in ambiguity_tags.items() if amb_type in tags}
        if not type_indices:
            continue

        nl_subset = [r for r in nl_results if r["idx"] in type_indices]
        dsl_subset = [r for r in dsl_results if r["idx"] in type_indices]

        nl_acc = compute_accuracy_at_k(nl_subset, max_rounds)
        dsl_acc = compute_accuracy_at_k(dsl_subset, max_rounds)

        nl_rounds = [
            r["solved_at_round"]
            for r in nl_subset
            if r.get("success") and r.get("solved_at_round") is not None
        ]
        avg_convergence = sum(nl_rounds) / max(1, len(nl_rounds)) if nl_rounds else None

        analysis["per_type"][amb_type] = {
            "count": len(type_indices),
            "nl_accuracy_at_k": nl_acc,
            "dsl_accuracy_at_k": dsl_acc,
            "gap_at_k": {k: dsl_acc[k] - nl_acc[k] for k in range(1, max_rounds + 1)},
            "nl_avg_convergence_round": avg_convergence,
        }

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nl-results", required=True, type=str)
    parser.add_argument("--dsl-results", required=True, type=str)
    parser.add_argument("--dataset-path", default="./data/test.jsonl", type=str)
    parser.add_argument("--max-rounds", default=5, type=int)
    parser.add_argument("--output", default="./result/analysis/", type=str)
    args = parser.parse_args()

    nl_results = load_jsonl(args.nl_results)
    dsl_results = load_jsonl(args.dsl_results)
    dataset = load_jsonl(args.dataset_path)

    analysis = analyze(nl_results, dsl_results, dataset, args.max_rounds)

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "ambiguity_analysis.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print("=== NL-DSL Gap at each round ===")
    for k, gap in analysis["gap_at_k"].items():
        print(f"  k={k}: {gap:+.4f}")

    print("\n=== Per-Ambiguity-Type Summary ===")
    for atype, data in analysis["per_type"].items():
        final_gap = data["gap_at_k"][args.max_rounds]
        conv = data["nl_avg_convergence_round"]
        conv_str = f"{conv:.1f}" if conv is not None else "N/A"
        print(f"  {atype} (n={data['count']}): gap@{args.max_rounds}={final_gap:+.4f}, avg_convergence={conv_str}")

    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
