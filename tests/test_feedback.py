from query4regex.verify.feedback import construct_feedback, FeedbackLevel
from query4regex.ops.op_dsl import Program, Op
from query4regex.regex.ast import Sym
from query4regex.config import DEFAULT_ALPHABET


def test_binary_feedback():
    fb = construct_feedback(
        level=FeedbackLevel.BINARY,
        gold_regex="a|b",
        predicted_regex="a",
        alphabet=DEFAULT_ALPHABET,
    )
    assert "incorrect" in fb.lower()
    assert "try again" in fb.lower()


def test_counterexample_feedback():
    fb = construct_feedback(
        level=FeedbackLevel.COUNTEREXAMPLE,
        gold_regex="a|b",
        predicted_regex="a",
        alphabet=DEFAULT_ALPHABET,
    )
    assert "b" in fb
    assert "incorrect" in fb.lower()


def test_diagnostic_feedback():
    inputs = {"r1": Sym("a"), "r2": Sym("b")}
    prog = Program(
        inputs=["r1", "r2"],
        ops=[
            Op(kind="UNION", args=["r1", "r2"]),
            Op(kind="STAR", args=["out"]),
        ],
    )
    fb = construct_feedback(
        level=FeedbackLevel.DIAGNOSTIC,
        gold_regex="(a|b)*",
        predicted_regex="a|b",
        alphabet=DEFAULT_ALPHABET,
        inputs=inputs,
        program=prog,
    )
    assert "step" in fb.lower()


def test_syntax_feedback():
    fb = construct_feedback(
        level=FeedbackLevel.BINARY,
        gold_regex="a|b",
        predicted_regex="",
        alphabet=DEFAULT_ALPHABET,
        is_unparsable=True,
    )
    assert "valid" in fb.lower() or "syntax" in fb.lower()
