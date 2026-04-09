from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Set, Optional
from pyformlang.finite_automaton import DeterministicFiniteAutomaton, Symbol
from ..eval.custom_equivalence import regex_to_nfa


@dataclass(frozen=True)
class Counterexample:
    string: str
    accepted_by: str   # "gold" or "predicted"
    rejected_by: str   # "gold" or "predicted"


def _dfa_from_regex_str(regex_str: str, alphabet: Set[str]) -> DeterministicFiniteAutomaton:
    nfa = regex_to_nfa(regex_str)
    dfa = nfa.to_deterministic().minimize()
    # Ensure alphabet is registered for complement operations
    for a in alphabet:
        dfa.add_symbol(Symbol(a))
    return dfa


def _find_accepting_path(dfa: DeterministicFiniteAutomaton) -> Optional[str]:
    if dfa.is_empty():
        return None
    start_states = list(dfa.start_states)
    if not start_states:
        return None

    queue = deque()
    visited = set()
    transitions = dfa.to_dict()

    for s in start_states:
        queue.append((s, ""))
        visited.add(s)

    while queue:
        state, path = queue.popleft()
        if state in dfa.final_states:
            return path
        if state in transitions:
            for sym, dst in transitions[state].items():
                if dst not in visited:
                    visited.add(dst)
                    queue.append((dst, path + str(sym)))
    return None


def find_counterexample(
    gold_regex: str,
    predicted_regex: str,
    alphabet: Set[str],
) -> Optional[Counterexample]:
    try:
        gold_dfa = _dfa_from_regex_str(gold_regex, alphabet)
        pred_dfa = _dfa_from_regex_str(predicted_regex, alphabet)
    except Exception:
        return None

    # gold accepts but predicted doesn't: gold ∩ ~predicted
    try:
        pred_compl = pred_dfa.get_complement()
        gold_minus_pred = gold_dfa.get_intersection(pred_compl).minimize()
        s = _find_accepting_path(gold_minus_pred)
        if s is not None:
            return Counterexample(string=s, accepted_by="gold", rejected_by="predicted")
    except Exception:
        pass

    # predicted accepts but gold doesn't: predicted ∩ ~gold
    try:
        gold_compl = gold_dfa.get_complement()
        pred_minus_gold = pred_dfa.get_intersection(gold_compl).minimize()
        s = _find_accepting_path(pred_minus_gold)
        if s is not None:
            return Counterexample(string=s, accepted_by="predicted", rejected_by="gold")
    except Exception:
        pass

    return None
