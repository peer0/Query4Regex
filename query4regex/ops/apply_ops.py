from __future__ import annotations
from typing import Dict, Set
from ..regex.ast import *
from ..regex.pretty import to_str
from ..regex.fragments import replace_by_index, replace_all_operators_of_kind, enumerate_operators
from .op_dsl import Program

def apply_ops(inputs: Dict[str, Regex], program: Program, alphabet: Set[str], allow_extended: bool = True) -> Regex:
    env: Dict[str, Regex] = dict(inputs)
    current: Regex | None = None

    def _bin_cls(kind:str):
        return {'UNION': UnionR, 'INTER': InterR, 'CONCAT': Concat}.get(kind)

    def _apply_unary(kind: str, node: Regex, args: list[str]) -> Regex:
        if kind == 'STAR':
            return Star(node)
        if kind == 'COMPL':
            return Compl(node)
        if kind == 'REVERSE':
            return Reverse(node)
        if kind == 'REPEAT':
            min_rep = int(args[1])
            max_rep = int(args[2]) if len(args) > 2 else None
            return Repeat(node, min_rep, max_rep)
        raise NotImplementedError(kind)

    for i, op in enumerate(program.ops):
        if i == 0:
            base = env[op.args[0]]
        else:
            base = current

        if op.kind in ('UNION', 'INTER', 'CONCAT'):
            r2 = env[op.args[1]]
            Bin = _bin_cls(op.kind)
            current = Bin(base, r2)
        elif op.kind in ('STAR', 'COMPL', 'REVERSE', 'REPEAT'):
            current = _apply_unary(op.kind, base, op.args)
        elif op.kind == 'REPLACE_OPERAND':
            idx = int(op.args[1])
            repl = env[op.args[2]]
            current = replace_by_index(base, idx, repl)
        elif op.kind == 'REPLACE_OPERATOR':
            op_nodes = enumerate_operators(base)
            idx = int(op.args[1])
            new_op_kind = op.args[2]
            if 0 <= idx < len(op_nodes):
                node_to_replace = op_nodes[idx]
                old_op_kind = "UNKNOWN"
                if isinstance(node_to_replace, Concat): old_op_kind = "CONCAT"
                elif isinstance(node_to_replace, UnionR): old_op_kind = "UNION"
                elif isinstance(node_to_replace, InterR): old_op_kind = "INTER"
                current = replace_all_operators_of_kind(base, old_op_kind, new_op_kind)
            else:
                current = base # No-op
        else:
            raise NotImplementedError(op.kind)

    return current