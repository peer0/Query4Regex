from __future__ import annotations
from typing import Set
from ..regex.parse import parse_regex_basic
from ..fa.automata import ast_to_dfa
from ..fa.equivalence import dfa_equivalent

def dfa_equal_acc(pred: str, gold: str, alphabet: Set[str]) -> bool:
    ap = ast_to_dfa(parse_regex_basic(pred), alphabet)
    ag = ast_to_dfa(parse_regex_basic(gold), alphabet)
    return dfa_equivalent(ap, ag)
