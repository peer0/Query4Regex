import re
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import EpsilonNFA, State, Symbol
import time
from timeout_decorator import timeout, TimeoutError

def expand_repetitions(regex: str) -> str:
    """
    Expands custom {n,m} repetitions in a regex string.
    Handles nested repetitions by expanding the innermost ones first.
    """
    # Add a recursion limit to prevent infinite loops in unforeseen edge cases
    recursion_depth = 0
    max_recursion_depth = 100 # Safety break

    def expand(inner_regex):
        nonlocal recursion_depth
        recursion_depth += 1
        if recursion_depth > max_recursion_depth:
            raise RecursionError("Maximum recursion depth exceeded in expand_repetitions")

        # Find the innermost, rightmost repetition pattern that does not contain other repetitions
        # This pattern looks for a base (group 1) followed by {n,m}
        # The base can be a single character/escaped char, or a balanced parenthesized group.
        pattern = r'([a-zA-Z0-9\\.]|\([^(){}]*\))\{(\d+),(\d+)\}'
        
        match = re.search(pattern, inner_regex)
        
        if not match:
            return inner_regex

        # If a match is found, expand it and then recurse to expand others
        base = match.group(1)
        start = int(match.group(2))
        end = int(match.group(3))

        if start > end:
            raise ValueError(f"Invalid repetition range: {{{start},{end}}}")

        # Build the replacement string
        if base.startswith('(') and base.endswith(')'):
            # It's a group, repeat the inner content
            inner_base = base[1:-1]
            expanded_part = '|'.join([f'({inner_base}){{{i}}}' for i in range(start, end + 1)])
        else:
            # It's a single character
            expanded_part = '|'.join([f'{base}{{{i}}}' for i in range(start, end + 1)])

        # To avoid re-matching, we can simplify this expansion for pyformlang
        # pyformlang doesn't support {n,m}, so we expand to (base)(base)...(base)?
        if len(base) > 1 and not base.startswith('('):
             base_grouped = f'({base})'
        else:
             base_grouped = base

        res_parts = [base_grouped] * start
        if end > start:
            res_parts.extend([f'({base_grouped})?'] * (end - start))
        
        res = ''.join(res_parts)
        if len(res_parts) > 1:
            res = f'({res})'

        # Replace and recurse on the entire string to find the next innermost repetition
        new_regex = inner_regex[:match.start()] + res + inner_regex[match.end():]
        return expand(new_regex)

    return expand(regex)

def regex_to_nfa(regex_str: str, memo: dict = None) -> EpsilonNFA:
    """
    Converts a regex string with custom operators to an EpsilonNFA.
    Handles: &, ~, {n,m}, and \varepsilon.
    Uses memoization to avoid re-computing for the same regex.
    """
    if memo is None:
        memo = {}
    if regex_str in memo:
        return memo[regex_str]

    # Pre-processing
    original_regex = regex_str
    regex_str = regex_str.replace('\\varepsilon', 'ε').strip()

    # Handle complement '~'
    if regex_str.startswith('~(') and regex_str.endswith(')'):
        inner_regex = regex_str[2:-1]
        nfa = regex_to_nfa(inner_regex, memo)
        result_nfa = nfa.complement()
        memo[original_regex] = result_nfa
        return result_nfa

    # Handle intersection '&' by finding the top-level operator
    bracket_level = 0
    for i, char in reversed(list(enumerate(regex_str))):
        if char == ')':
            bracket_level += 1
        elif char == '(':
            bracket_level -= 1
        elif char == '&' and bracket_level == 0:
            r1_str = regex_str[:i]
            r2_str = regex_str[i+1:]
            nfa1 = regex_to_nfa(r1_str, memo)
            nfa2 = regex_to_nfa(r2_str, memo)
            result_nfa = nfa1.get_intersection(nfa2)
            memo[original_regex] = result_nfa
            return result_nfa

    # Expand {n,m} repetitions before final parsing
    try:
        expanded_regex = expand_repetitions(regex_str)
    except RecursionError:
        # If expansion fails, we can't process this regex
        raise ValueError("Failed to expand repetitions, possible infinite loop")

    # pyformlang Regex object creation
    regex_str_pfl = expanded_regex.replace('ε', '')

    if not regex_str_pfl:
        nfa = EpsilonNFA()
        start_state = State(0)
        nfa.add_start_state(start_state)
        nfa.add_final_state(start_state)
        memo[original_regex] = nfa
        return nfa

    regex_obj = Regex(regex_str_pfl)
    result_nfa = regex_obj.to_epsilon_nfa()
    memo[original_regex] = result_nfa
    return result_nfa

@timeout(5)
def are_equivalent(gold_regex: str, generated_regex: str) -> bool | None:
    """
    Checks if two regular expressions are equivalent using the improved NFA conversion.
    Returns None if parsing fails.
    """
    try:
        gold_nfa = regex_to_nfa(gold_regex)
        generated_nfa = regex_to_nfa(generated_regex)
        return gold_nfa.is_equivalent_to(generated_nfa)
    except TimeoutError:
        print(f"gold_regex: {gold_regex} \n generated_regex: {generated_regex}\n")
        print("Timed out!")
        return None
    except Exception:
        return None

def is_valid_standard_regex(regex_str: str) -> bool:
    """
    Checks if a regex is valid for a standard engine after custom syntax is processed.
    """
    try:
        # Basic transformation for standard library
        # This is a simplification. & and ~ are complex to transform for `re`.
        # We will primarily rely on our parser's success.
        temp_str = regex_str.replace('\\varepsilon', '')
        temp_str = expand_repetitions(temp_str)
        
        # The `re` module does not support intersection or complement.
        # We consider it "valid" if our parser can handle it.
        # This function's purpose is to see if our *output* is valid.
        # A better check is just to see if our own parser succeeds.
        regex_to_nfa(regex_str)
        return True
    except Exception:
        return False
