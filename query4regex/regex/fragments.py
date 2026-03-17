from .ast import *

def enumerate_operators(r: Regex) -> list[Regex]:
    ops: list[Regex] = []
    def visit(x: Regex) -> None:
        if isinstance(x, (Concat, UnionR, InterR)):
            ops.append(x)
            visit(x.left); visit(x.right)
        elif isinstance(x, (Star, Compl, Reverse, Repeat)):
            ops.append(x)
            visit(x.inner)
        else:
            return
    visit(r)
    return ops

def replace_all_operators_of_kind(r: Regex, old_kind: str, new_kind: str) -> Regex:
    """
    Recursively traverses a Regex AST and replaces all operators of a
    given kind with a new kind.
    """
    def _get_op_class(kind_str: str):
        return {
            "UNION": UnionR, "INTER": InterR, "CONCAT": Concat,
            "STAR": Star, "COMPL": Compl, "REVERSE": Reverse,
        }.get(kind_str)

    OldOp = _get_op_class(old_kind)
    NewOp = _get_op_class(new_kind)

    if OldOp is None or NewOp is None:
        return r # Invalid op kinds

    def _replace(node: Regex) -> Regex:
        # Recurse first
        if isinstance(node, (Concat, UnionR, InterR)):
            new_left = _replace(node.left)
            new_right = _replace(node.right)
            # Check if the current node should be replaced
            if isinstance(node, OldOp):
                return NewOp(new_left, new_right)
            return node.__class__(new_left, new_right)

        elif isinstance(node, (Star, Compl, Reverse, Repeat)):
            new_inner = _replace(node.inner)
            # Check if the current node should be replaced
            if isinstance(node, OldOp):
                return NewOp(new_inner)
            if isinstance(node, Repeat):
                return Repeat(new_inner, node.min, node.max)
            return node.__class__(new_inner)
        
        return node # Atomic nodes

    return _replace(r)


def enumerate_fragments(r: Regex) -> list[Regex]:
    res: list[Regex] = []
    def visit(x: Regex):
        res.append(x)
        if isinstance(x, (Concat, UnionR, InterR)):
            visit(x.left); visit(x.right)
        elif isinstance(x, (Star, Compl, Reverse, Repeat)):
            visit(x.inner)
    visit(r)
    return res

def swap_by_index(r: Regex, i: int, j: int) -> Regex:
    frags = enumerate_fragments(r)
    if i < 0 or j < 0 or i >= len(frags) or j >= len(frags) or i == j:
        return r

    frag_i = frags[i]
    frag_j = frags[j]

    # Use a unique placeholder that is not in the regex
    placeholder_sym = "___PLACEHOLDER___"
    
    # Ensure placeholder is not in the regex
    all_syms = [s.sym for s in frags if isinstance(s, Sym)]
    while placeholder_sym in all_syms:
        placeholder_sym += "_"

    def _replace(node: Regex, target: Regex, replacement: Regex) -> Regex:
        if node == target:
            return replacement
        if isinstance(node, (Concat, UnionR, InterR)):
            return node.__class__(_replace(node.left, target, replacement),
                                  _replace(node.right, target, replacement))
        elif isinstance(node, (Star, Compl, Reverse, Repeat)):
            if isinstance(node, Repeat):
                return node.__class__(_replace(node.inner, target, replacement), node.min, node.max)
            else:
                return node.__class__(_replace(node.inner, target, replacement))
        return node

    # To handle cases where one fragment is a sub-tree of the other,
    # we need to check for this and swap the order of replacement.
    if frag_j in enumerate_fragments(frag_i):
        frag_i, frag_j = frag_j, frag_i

    placeholder = Sym(placeholder_sym)
    r_with_placeholder = _replace(r, frag_i, placeholder)
    r_swapped_j = _replace(r_with_placeholder, frag_j, frag_i)
    r_final = _replace(r_swapped_j, placeholder, frag_j)

    return r_final

def replace_by_index(r: Regex, i: int, new_frag: Regex) -> Regex:
    frags = enumerate_fragments(r)
    if i < 0 or i >= len(frags):
        return r

    frag_to_replace = frags[i]

    def _replace(node: Regex, target: Regex, replacement: Regex) -> Regex:
        if node == target:
            return replacement
        if isinstance(node, (Concat, UnionR, InterR)):
            return node.__class__(_replace(node.left, target, replacement),
                                  _replace(node.right, target, replacement))
        elif isinstance(node, (Star, Compl, Reverse, Repeat)):
            if isinstance(node, Repeat):
                return node.__class__(_replace(node.inner, target, replacement), node.min, node.max)
            else:
                return node.__class__(_replace(node.inner, target, replacement))
        return node

    return _replace(r, frag_to_replace, new_frag)