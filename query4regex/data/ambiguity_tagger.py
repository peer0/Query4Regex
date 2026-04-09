from __future__ import annotations
import re
from typing import List


def tag_ambiguity(instruction: str, ops: List[str]) -> List[str]:
    tags: List[str] = []
    lower = instruction.lower()
    is_multi_step = len(ops) > 1

    # Scope ambiguity
    if is_multi_step and re.search(r'\b(the result|the whole result|the entire result)\b', lower):
        tags.append("scope")

    # Anaphoric reference
    if is_multi_step and re.search(r'(then|,)\s.*\b(it|this|that)\b', lower):
        tags.append("anaphoric")

    # Operator scope
    if is_multi_step:
        binary = {"UNION", "INTER", "CONCAT"}
        unary = {"STAR", "COMPL", "REVERSE", "REPEAT"}
        for i in range(1, len(ops)):
            if ops[i] in unary and ops[i - 1] in binary:
                if "then" not in lower:
                    tags.append("operator_scope")
                    break

    # Implicit ordering
    op_keywords = ["union", "intersect", "complement", "concatenat", "star", "repeat", "reverse"]
    found_ops = [kw for kw in op_keywords if kw in lower]
    if len(found_ops) >= 2 and " and " in lower and "then" not in lower:
        tags.append("implicit_ordering")

    # Fragment reference
    if re.search(r'\b(part|portion|fragment|section|element)\b', lower):
        tags.append("fragment")
    if any(op in ("REPLACE_OPERAND", "REPLACE_OPERATOR") for op in ops):
        if "fragment" not in tags:
            tags.append("fragment")

    return tags
