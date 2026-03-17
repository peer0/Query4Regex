from .ast import *
from .simplify import reverse_ast

def _paren(x: Regex, allow_extended=True) -> str:
    if isinstance(x, Sym): return x.sym
    if isinstance(x, Epsilon): return 'ε'
    if isinstance(x, Empty): return '∅'
    s = to_str(x, allow_extended=allow_extended)
    return f'({s})'

def to_str(r: Regex, allow_extended: bool = True) -> str:
    if isinstance(r, Epsilon): return 'ε'
    if isinstance(r, Empty): return '∅'
    if isinstance(r, Sym): return r.sym
    if isinstance(r, Star): return f'({_paren(r.inner, allow_extended)})*'
    if isinstance(r, Reverse):
        reversed_r = reverse_ast(r.inner)
        if reversed_r == r.inner: # Not reversible by this method
            return f'rev({_paren(r.inner, allow_extended)})'
        return to_str(reversed_r, allow_extended)
    if isinstance(r, Compl):
        return f'~({_paren(r.inner, allow_extended)})' if allow_extended else f'COMPL({to_str(r.inner, True)})'
    if isinstance(r, Concat): return f'{_paren(r.left)}{_paren(r.right)}'
    if isinstance(r, UnionR): return f'{_paren(r.left)}|{_paren(r.right)}'
    if isinstance(r, InterR):
        return f'{_paren(r.left)}&{_paren(r.right)}' if allow_extended else f'INTER({to_str(r.left, True)},{to_str(r.right, True)})'
    if isinstance(r, Repeat):
        if r.max is None:
            return f'{_paren(r.inner, allow_extended)}{{{r.min},}}'
        return f'{_paren(r.inner, allow_extended)}{{{r.min},{r.max}}}'
    raise TypeError(r)