#!/usr/bin/env python3
import argparse
from query4regex.config import make_alphabet, DEFAULT_ALPHABET
from query4regex.eval.evaluate import evaluate_file

def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate predictions vs. gold JSONL")
    p.add_argument("--preds", required=True)
    p.add_argument("--gold", required=True)
    p.add_argument("--alphabet", nargs="+", default=list(DEFAULT_ALPHABET))
    args = p.parse_args(argv)

    acc = evaluate_file(args.preds, args.gold, make_alphabet(args.alphabet))
    print(f"semantic_accuracy={acc:.4f}")

if __name__ == "__main__":
    main()
