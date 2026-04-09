from query4regex.verify.intermediate import compute_gold_intermediates, detect_divergence
from query4regex.ops.op_dsl import Program, Op
from query4regex.regex.ast import Sym
from query4regex.config import DEFAULT_ALPHABET


def test_compute_intermediates_two_steps():
    inputs = {"r1": Sym("a"), "r2": Sym("b")}
    prog = Program(
        inputs=["r1", "r2"],
        ops=[
            Op(kind="UNION", args=["r1", "r2"]),
            Op(kind="STAR", args=["out"]),
        ],
    )
    intermediates = compute_gold_intermediates(inputs, prog, DEFAULT_ALPHABET)
    assert len(intermediates) == 2
    assert intermediates[0].step_index == 0
    assert intermediates[0].op_kind == "UNION"
    assert intermediates[1].step_index == 1
    assert intermediates[1].op_kind == "STAR"


def test_detect_divergence_partial_correctness():
    inputs = {"r1": Sym("a"), "r2": Sym("b")}
    prog = Program(
        inputs=["r1", "r2"],
        ops=[
            Op(kind="UNION", args=["r1", "r2"]),
            Op(kind="STAR", args=["out"]),
        ],
    )
    intermediates = compute_gold_intermediates(inputs, prog, DEFAULT_ALPHABET)
    divergence = detect_divergence("a|b", intermediates, DEFAULT_ALPHABET)
    assert divergence is not None
    assert divergence.matches_step == 0
    assert divergence.diverges_at_step == 1
    assert divergence.diagnosis == "partial_correctness"


def test_detect_divergence_fully_correct():
    inputs = {"r1": Sym("a"), "r2": Sym("b")}
    prog = Program(
        inputs=["r1", "r2"],
        ops=[
            Op(kind="UNION", args=["r1", "r2"]),
            Op(kind="STAR", args=["out"]),
        ],
    )
    intermediates = compute_gold_intermediates(inputs, prog, DEFAULT_ALPHABET)
    divergence = detect_divergence("(a|b)*", intermediates, DEFAULT_ALPHABET)
    assert divergence is None
