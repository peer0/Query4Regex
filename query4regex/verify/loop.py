from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set
from ..eval.custom_equivalence import are_equivalent
from ..ops.op_dsl import Program, Op
from ..regex.ast import Regex
from ..regex.parse import parse_regex_basic
from .feedback import FeedbackLevel, construct_feedback


class GenerativeModel(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class RoundRecord:
    round_num: int
    predicted_regex: str
    parsable: bool
    equivalent: Optional[bool]
    feedback: str


@dataclass
class VerificationResult:
    success: bool
    solved_at_round: Optional[int]
    rounds: List[RoundRecord] = field(default_factory=list)


def _extract_regex(answer: str) -> str:
    match = re.search(r'oxed\{(.*)\}', answer, re.DOTALL)
    return match.group(1) if match else ""


def _is_parsable(regex_str: str) -> bool:
    if not regex_str:
        return False
    try:
        parse_regex_basic(regex_str)
        return True
    except Exception:
        return False


def _build_prompt(
    inputs: Dict[str, str],
    instruction: str,
    ops_dsl: str,
    pipeline: str,
    history: List[RoundRecord],
) -> str:
    regex_inputs = "\n".join(f"{name}: {regex}" for name, regex in inputs.items())

    if pipeline == "nl2dsl2regex":
        base = (
            f"Given the following regular expressions:\n{regex_inputs}\n\n"
            f"Instruction: {instruction}\n\nOps_dsl: {ops_dsl}\n\nResulting regex:"
        )
    else:
        base = (
            f"Given the following regular expressions:\n{regex_inputs}\n\n"
            f"Instruction: {instruction}\n\nResulting regex:"
        )

    if not history:
        return base

    parts = [base]
    for rec in history:
        parts.append(f" \\boxed{{{rec.predicted_regex}}}")
        parts.append(f"\n\nFeedback: {rec.feedback}\n\nRevised regex:")
    return "".join(parts)


def _parse_program_from_dsl(ops_dsl: str) -> Optional[Program]:
    try:
        header, body = ops_dsl.split("::", 1)
        import ast as stdlib_ast
        inputs_str = header.strip().replace("inputs=", "")
        input_names = stdlib_ast.literal_eval(inputs_str)

        steps = [s.strip() for s in body.strip().split(";")]
        ops = []
        for step in steps:
            op_part = step.split("->")[0].strip()
            match = re.match(r'(\w+)\(([^)]*)\)', op_part)
            if match:
                kind = match.group(1)
                args = [a.strip() for a in match.group(2).split(",")]
                ops.append(Op(kind=kind, args=args))
        return Program(inputs=input_names, ops=ops)
    except Exception:
        return None


def run_verification_loop(
    model: GenerativeModel,
    gold_regex: str,
    inputs: Dict[str, str],
    instruction: str,
    ops_dsl: str,
    pipeline: str,
    alphabet: Set[str],
    feedback_level: FeedbackLevel,
    max_rounds: int = 5,
    input_asts: Optional[Dict[str, Regex]] = None,
) -> VerificationResult:
    history: List[RoundRecord] = []

    parsed_inputs: Optional[Dict[str, Regex]] = input_asts
    parsed_program: Optional[Program] = None
    if feedback_level == FeedbackLevel.DIAGNOSTIC:
        if parsed_inputs is None:
            try:
                parsed_inputs = {k: parse_regex_basic(v) for k, v in inputs.items()}
            except Exception:
                parsed_inputs = None
        parsed_program = _parse_program_from_dsl(ops_dsl)

    for round_num in range(1, max_rounds + 1):
        prompt = _build_prompt(inputs, instruction, ops_dsl, pipeline, history)
        answer = model.generate(prompt)
        predicted = _extract_regex(answer)

        parsable = _is_parsable(predicted)

        if not parsable:
            fb = construct_feedback(
                level=feedback_level,
                gold_regex=gold_regex,
                predicted_regex=predicted,
                alphabet=alphabet,
                is_unparsable=True,
            )
            history.append(RoundRecord(
                round_num=round_num,
                predicted_regex=predicted,
                parsable=False,
                equivalent=None,
                feedback=fb,
            ))
            continue

        equiv = are_equivalent(gold_regex, predicted)
        if equiv is True:
            history.append(RoundRecord(
                round_num=round_num,
                predicted_regex=predicted,
                parsable=True,
                equivalent=True,
                feedback="",
            ))
            return VerificationResult(
                success=True, solved_at_round=round_num, rounds=history
            )

        fb = construct_feedback(
            level=feedback_level,
            gold_regex=gold_regex,
            predicted_regex=predicted,
            alphabet=alphabet,
            inputs=parsed_inputs,
            program=parsed_program,
        )
        history.append(RoundRecord(
            round_num=round_num,
            predicted_regex=predicted,
            parsable=True,
            equivalent=False,
            feedback=fb,
        ))

    return VerificationResult(success=False, solved_at_round=None, rounds=history)
