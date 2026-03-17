from .ast import *

def simplify(r: Regex) -> Regex:
    # Tiny set of examples; extend as needed.
    if isinstance(r, UnionR) and isinstance(r.left, Empty): return r.right
    if isinstance(r, UnionR) and isinstance(r.right, Empty): return r.left
    if isinstance(r, Concat) and isinstance(r.left, Epsilon): return r.right
    if isinstance(r, Concat) and isinstance(r.right, Epsilon): return r.left
    if isinstance(r, Star) and isinstance(r.inner, Star): return r.inner
    if isinstance(r, Repeat) and r.min == 1 and r.max == 1: return r.inner
    return r

def reverse_ast(r: Regex) -> Regex:
    if isinstance(r, (Sym, Epsilon, Empty)):
        return r
    if isinstance(r, Concat):
        return Concat(reverse_ast(r.right), reverse_ast(r.left))
    if isinstance(r, UnionR):
        return UnionR(reverse_ast(r.left), reverse_ast(r.right))
    if isinstance(r, Star):
        return Star(reverse_ast(r.inner))
    if isinstance(r, Repeat):
        return Repeat(reverse_ast(r.inner), r.min, r.max)
    if isinstance(r, Reverse):
        return r.inner
    if isinstance(r, (InterR, Compl)):
        return r # Cannot be distributed, return unchanged

    raise TypeError(f"Unsupported regex type for reversal: {type(r)}")