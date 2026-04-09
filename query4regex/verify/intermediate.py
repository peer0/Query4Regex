from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from ..regex.ast import Regex
from ..regex.pretty import to_str
from ..ops.op_dsl import Program
from ..ops.apply_ops import apply_ops as _full_apply_ops
from ..eval.custom_equivalence import regex_to_nfa


@dataclass(frozen=True)
class StepResult:
    step_index: int
    op_kind: str
    regex_str: str
    regex_ast: Regex


@dataclass(frozen=True)
class Divergence:
    matches_step: int
    diverges_at_step: int
    diagnosis: str
    description: str


def compute_gold_intermediates(
    inputs: Dict[str, Regex],
    program: Program,
    alphabet: Set[str],
) -> List[StepResult]:
    results = []
    for i in range(len(program.ops)):
        partial_prog = Program(
            inputs=program.inputs,
            ops=program.ops[: i + 1],
            output_alias=program.output_alias,
        )
        partial_ast = _full_apply_ops(inputs, partial_prog, alphabet)
        results.append(
            StepResult(
                step_index=i,
                op_kind=program.ops[i].kind,
                regex_str=to_str(partial_ast, allow_extended=True),
                regex_ast=partial_ast,
            )
        )
    return results


def _are_equivalent_safe(regex_str: str, step: StepResult, alphabet: Set[str]) -> bool:
    try:
        pred_nfa = regex_to_nfa(regex_str)
        gold_nfa = regex_to_nfa(step.regex_str)
        return pred_nfa.is_equivalent_to(gold_nfa)
    except Exception:
        return False


def detect_divergence(
    predicted_regex: str,
    intermediates: List[StepResult],
    alphabet: Set[str],
) -> Optional[Divergence]:
    if not intermediates:
        return None

    final = intermediates[-1]
    if _are_equivalent_safe(predicted_regex, final, alphabet):
        return None

    matches_step = -1
    for step in intermediates[:-1]:
        if _are_equivalent_safe(predicted_regex, step, alphabet):
            matches_step = step.step_index

    diverges_at = matches_step + 1

    if matches_step >= 0:
        return Divergence(
            matches_step=matches_step,
            diverges_at_step=diverges_at,
            diagnosis="partial_correctness",
            description=(
                f"Your output matches the result after step {matches_step + 1} "
                f"({intermediates[matches_step].op_kind}), but diverges at step "
                f"{diverges_at + 1} ({intermediates[diverges_at].op_kind})."
            ),
        )

    return Divergence(
        matches_step=-1,
        diverges_at_step=0,
        diagnosis="unknown",
        description=(
            f"Your output does not match any intermediate step. "
            f"The first operation is {intermediates[0].op_kind}."
        ),
    )
