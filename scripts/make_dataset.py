#!/usr/bin/env python3
import argparse
from typing import List
from query4regex.config import make_alphabet, DEFAULT_ALPHABET
from query4regex.data.synth_pipeline import generate_corpus

def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate a synthetic Query4Regex corpus")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--alphabet", nargs="+", default=list(DEFAULT_ALPHABET))
    args = p.parse_args(argv)

    alphabet = make_alphabet(args.alphabet)
    generate_corpus(args.n, args.out, alphabet=alphabet)

if __name__ == "__main__":
    main()
