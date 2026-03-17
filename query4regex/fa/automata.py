from __future__ import annotations
from typing import Set
from pyformlang.regular_expression import Regex as PFRegex
from pyformlang.finite_automaton import EpsilonNFA, DeterministicFiniteAutomaton, NondeterministicFiniteAutomaton, State, Symbol
from .equivalence import dfa_minimize
from ..regex.ast import *

def _ast_to_pfregex_str(r: Regex) -> str:
    if isinstance(r, Sym): return r.sym
    if isinstance(r, Epsilon): return '$'
    if isinstance(r, Empty): return '#'
    if isinstance(r, Star): return f'({_ast_to_pfregex_str(r.inner)})*'
    if isinstance(r, Concat): return f'({_ast_to_pfregex_str(r.left)})({_ast_to_pfregex_str(r.right)})'
    if isinstance(r, UnionR): return f'({_ast_to_pfregex_str(r.left)})|({_ast_to_pfregex_str(r.right)})'
    if isinstance(r, Repeat):
        if r.max is None:
            return f'({_ast_to_pfregex_str(r.inner)}){{{r.min},}}'
        return f'({_ast_to_pfregex_str(r.inner)}){{{r.min},{r.max}}}'
    raise NotImplementedError('Use DFA ops for Inter/Compl/Reverse')

def reverse_dfa(dfa: DeterministicFiniteAutomaton) -> DeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()
    # pyformlang's DFA isn't directly iterable; use to_dict()
    for (src, sym), dsts in dfa.to_dict().items():
        for dst in dsts:
            if sym is None:
                nfa.add_epsilon_transition(State(dst), State(src))
            else:
                nfa.add_transition(State(dst), Symbol(sym.value), State(src))
    for s in dfa.final_states: nfa.add_start_state(State(s.value))
    for s in dfa.start_states: nfa.add_final_state(State(s.value))
    return nfa.to_deterministic().minimize()

def ast_to_dfa(r: Regex, alphabet: Set[str]) -> DeterministicFiniteAutomaton:
    if isinstance(r, Compl):
        base = ast_to_dfa(r.inner, alphabet)
        return base.get_complement(alphabet={Symbol(a) for a in alphabet}).minimize()
    if isinstance(r, InterR):
        left = ast_to_dfa(r.left, alphabet)
        right = ast_to_dfa(r.right, alphabet)
        return left.get_intersection(right).minimize()
    if isinstance(r, Reverse):
        dfa = ast_to_dfa(r.inner, alphabet)
        return reverse_dfa(dfa)
    enfa: EpsilonNFA = PFRegex(_ast_to_pfregex_str(r)).to_epsilon_nfa()
    return enfa.to_deterministic().minimize()
