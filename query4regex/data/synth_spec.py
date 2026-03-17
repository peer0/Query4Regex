from __future__ import annotations
import random
from typing import List, Set
from ..regex.ast import *

def _sample_atom(alphabet: Set[str]) -> Regex:
    ch = random.choice(sorted(list(alphabet)))
    return Sym(ch)

def _sample_regex(depth: int, alphabet: Set[str]) -> Regex:
    if depth <= 0:
        return random.choice([Epsilon(), _sample_atom(alphabet)])
    choice = random.choice(['sym','union','concat','star', 'repeat'])
    if choice == 'sym':
        return _sample_atom(alphabet)
    if choice == 'union':
        return UnionR(_sample_regex(depth-1, alphabet), _sample_regex(depth-1, alphabet))
    if choice == 'concat':
        return Concat(_sample_regex(depth-1, alphabet), _sample_regex(depth-1, alphabet))
    if choice == 'star':
        return Star(_sample_regex(depth-1, alphabet))
    if choice == 'repeat':
        min_val = random.randint(0, 5)
        max_val = random.randint(min_val, min_val + 5) if random.random() > 0.5 else None
        return Repeat(_sample_regex(depth-1, alphabet), min_val, max_val)
    return _sample_atom(alphabet)

def sample_base_regexes(k: int, max_depth: int, alphabet: Set[str]) -> List[Regex]:
    return [_sample_regex(max_depth, alphabet) for _ in range(k)]
