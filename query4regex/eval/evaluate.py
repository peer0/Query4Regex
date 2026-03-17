import json
from typing import Iterable, Set, Tuple
from .metrics import dfa_equal_acc

def evaluate_predictions(preds: Iterable[Tuple[str, str]], alphabet: Set[str]) -> float:
    total = 0
    correct = 0
    for p, g in preds:
        total += 1
        if dfa_equal_acc(p, g, alphabet):
            correct += 1
    return correct / max(1, total)

def evaluate_file(pred_path: str, gold_path: str, alphabet: Set[str]) -> float:
    preds = []
    with open(pred_path, 'r', encoding='utf-8') as pf, open(gold_path, 'r', encoding='utf-8') as gf:
        gold = [json.loads(l) for l in gf]
        for i, line in enumerate(pf):
            p = line.strip()
            g = gold[i]['gold_regex']
            preds.append((p, g))
    return evaluate_predictions(preds, alphabet)
