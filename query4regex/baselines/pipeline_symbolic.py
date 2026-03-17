from __future__ import annotations
from typing import Dict, Set
from ..nl.parse_instruction import parse_instruction
from ..regex.parse import parse_regex_basic
from ..ops.apply_ops import apply_ops
from ..regex.pretty import to_str

def run_symbolic(nl: str, inputs: Dict[str, str], alphabet: Set[str]) -> str:
    ast_inputs = {k: parse_regex_basic(v) for k, v in inputs.items()}
    prog = parse_instruction(nl, list(inputs.keys()))
    out_ast = apply_ops(ast_inputs, prog, alphabet, allow_extended=True)
    return to_str(out_ast, allow_extended=True)
