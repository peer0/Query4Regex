import re
from typing import List
from .paraphrase import normalize
from ..ops.op_dsl import Op, Program

def parse_instruction(nl: str, input_aliases: List[str]) -> Program:
    """
    Parse a natural language instruction into a Program consisting of a sequence of Ops.

    This parser is intentionally heuristic‑driven and supports multiple operations in a single
    instruction.  It splits the normalized instruction on occurrences of the word "then"
    and attempts to extract exactly one operation per segment.  Each extracted operation
    records the alias(es) referenced in that segment when present; if none are explicitly
    mentioned, the parser falls back to the first input alias.  When a binary operation
    is identified and only one alias is present in the segment, the parser assumes the
    two input aliases provided to the function.

    Examples:

      "Take the union of r1 and r2, then apply Kleene star to r2" →
        [UNION(r1,r2), STAR(r2)]

      "Compute the intersection of r1 and r2 then reverse r1" →
        [INTER(r1,r2), REVERSE(r1)]

    Note: this parser does not attempt to resolve the scope of intermediate results; it
    simply forwards the specified aliases into each op.  The semantics of sequential
    application (e.g. whether an op should act on the previous result or on a fresh
    input) is handled by `apply_ops`, which always binds the result of each op to
    `program.output_alias` (defaulting to "out").
    """

    text = normalize(nl)
    ops: List[Op] = []

    # Split on 'then' to support sequences like "..., then ...".
    segments = re.split(r"\bthen\b", text)

    for segment in segments:
        seg = segment.strip(" ,.; ")
        if not seg:
            continue

        # Capture explicit alias references (e.g. r1, r2) in this segment.
        alias_refs = re.findall(r"\br(\d+)\b", seg)
        aliases = [f"r{n}" for n in alias_refs]

        # Helper to choose aliases for binary ops.
        def binary_args() -> List[str]:
            if len(aliases) >= 2:
                return aliases[:2]
            return input_aliases[:2]

        # Helper for unary ops.
        def unary_arg() -> List[str]:
            if aliases:
                return [aliases[0]]
            return [input_aliases[0]]

        # Fragment swap e.g. "swap f1 and f3"
        m_swap = re.search(r"(?:swap|exchange|reorder)\s+f(\d+)\s+(?:and|with)\s+f(\d+)", seg)
        if m_swap:
            base_alias = aliases[0] if aliases else input_aliases[0]
            ops.append(Op(kind="SWAP_FRAG", args=[base_alias, m_swap.group(1), m_swap.group(2)]))
            continue

        # Repeat operation: "star m to n times"
        m_repeat = re.search(r"\bstar\s+(\d+)\s+to\s+(\d+)\s+times", seg)
        if m_repeat:
            base_alias = aliases[0] if aliases else input_aliases[0]
            ops.append(Op(kind="REPEAT", args=[base_alias, m_repeat.group(1), m_repeat.group(2)]))
            continue

        # Swap operands/arguments
        if re.search(r"swap\s+(?:operands|arguments)", seg):
            base_alias = aliases[0] if aliases else input_aliases[0]
            ops.append(Op(kind="REPLACE_OPERAND", args=[base_alias]))
            continue

        # Replace operator
        m_replace_op = re.search(r"(?:change|replace)\s+all\s+(?:occurrences\s+of\s+)?(UNION|INTER|CONCAT|STAR|COMPL|REVERSE)\s+(?:operators\s+to|with)\s+(UNION|INTER|CONCAT|STAR|COMPL|REVERSE)", seg, re.IGNORECASE)
        if m_replace_op:
            base_alias = aliases[0] if aliases else input_aliases[0]
            old_op, new_op = m_replace_op.groups()
            ops.append(Op(kind="REPLACE_OPERATOR", args=[base_alias, old_op.upper(), new_op.upper()]))
            continue

        # Intersection / both / and
        if re.search(r"\b(intersection)\b", seg):
            ops.append(Op(kind="INTER", args=binary_args()))
            continue

        # Union / either / or
        if re.search(r"\b(union)\b", seg):
            ops.append(Op(kind="UNION", args=binary_args()))
            continue

        # Complement / not / exclude / excluding
        if re.search(r"\b(complement|not |exclude |excluding )\b", seg):
            ops.append(Op(kind="COMPL", args=unary_arg()))
            continue

        # Concatenation / followed by
        if re.search(r"\b(concat|concatenate )\b", seg):
            ops.append(Op(kind="CONCAT", args=binary_args()))
            continue

        # Kleene star / zero or more / repetition / repeat / star
        if re.search(r"\b(kleene star|zero or more |repetition |repeat |star)\b", seg):
            ops.append(Op(kind="STAR", args=unary_arg()))
            continue

        # Reverse
        if re.search(r"\b(reverse|reversal |reversed )\b", seg):
            ops.append(Op(kind="REVERSE", args=unary_arg()))
            continue

        # No pattern matched for this segment
        raise ValueError(f"Unable to parse segment '{seg}' in instruction: {nl}")

    if not ops:
        raise ValueError(f"Unable to parse instruction: {nl}")

    return Program(inputs=input_aliases, ops=ops)

