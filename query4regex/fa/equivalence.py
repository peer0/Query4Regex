from pyformlang.finite_automaton import DeterministicFiniteAutomaton

def dfa_minimize(dfa: DeterministicFiniteAutomaton) -> DeterministicFiniteAutomaton:
    return dfa.minimize()

def dfa_equivalent(a: DeterministicFiniteAutomaton, b: DeterministicFiniteAutomaton) -> bool:
    return a.get_symmetric_difference(b).is_empty()
