from __future__ import annotations
from enum import Enum
from typing import Dict, Optional, Set
from ..regex.ast import Regex
from ..ops.op_dsl import Program
from .counterexample import find_counterexample
from .intermediate import compute_gold_intermediates, detect_divergence


class FeedbackLevel(Enum):
    BINARY = "binary"
    COUNTEREXAMPLE = "counterexample"
    DIAGNOSTIC = "diagnostic"


_BINARY_MSG = "Your answer is incorrect. Try again."
_SYNTAX_MSG = "Your output is not a syntactically valid regex. Please provide a valid regular expression."


def construct_feedback(
    level: FeedbackLevel,
    gold_regex: str,
    predicted_regex: str,
    alphabet: Set[str],
    inputs: Optional[Dict[str, Regex]] = None,
    program: Optional[Program] = None,
    is_unparsable: bool = False,
) -> str:
    if is_unparsable:
        return _SYNTAX_MSG

    if level == FeedbackLevel.BINARY:
        return _BINARY_MSG

    ce = find_counterexample(gold_regex, predicted_regex, alphabet)

    ce_msg = ""
    if ce is not None:
        if ce.accepted_by == "predicted":
            ce_msg = (
                f" The string \"{ce.string}\" is accepted by your regex "
                f"but should be rejected by the target."
            )
        else:
            ce_msg = (
                f" The string \"{ce.string}\" should be accepted by the "
                f"target but your regex rejects it."
            )

    if level == FeedbackLevel.COUNTEREXAMPLE:
        return f"Your regex `{predicted_regex}` is incorrect.{ce_msg}" if ce_msg else _BINARY_MSG

    # Level C: DIAGNOSTIC
    diag_msg = ""
    if inputs is not None and program is not None:
        try:
            intermediates = compute_gold_intermediates(inputs, program, alphabet)
            divergence = detect_divergence(predicted_regex, intermediates, alphabet)
            if divergence is not None:
                diag_msg = f" {divergence.description}"
        except Exception:
            pass

    base = f"Your regex `{predicted_regex}` is incorrect.{ce_msg}"
    if diag_msg:
        return f"{base}{diag_msg}"
    if ce_msg:
        return base
    return _BINARY_MSG
