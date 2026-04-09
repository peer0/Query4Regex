#!/usr/bin/env python3
"""Add ambiguity tags to an existing dataset file."""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query4regex.data.ambiguity_tagger import tag_ambiguity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", default=None, type=str)
    args = parser.parse_args()

    output = args.output
    if output is None:
        base, ext = os.path.splitext(args.input)
        output = f"{base}_tagged{ext}"

    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    for rec in records:
        ops = rec.get("meta", {}).get("ops", [])
        instruction = rec.get("instruction", "")
        rec["meta"]["ambiguity_tags"] = tag_ambiguity(instruction, ops)

    with open(output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    all_tags = []
    for rec in records:
        all_tags.extend(rec["meta"]["ambiguity_tags"])
    counts = Counter(all_tags)
    print(f"Tagged {len(records)} instances -> {output}")
    print(f"Tag distribution: {dict(counts)}")


if __name__ == "__main__":
    main()
