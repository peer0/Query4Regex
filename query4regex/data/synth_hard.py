from __future__ import annotations
import json
import random
import time
from typing import Dict, Optional, Set
from ..config import DEFAULT_ALPHABET
from ..regex.ast import Regex
from ..regex.pretty import to_str
from ..ops.op_dsl import Program, Op
from ..ops.apply_ops import apply_ops
from .synth_spec import sample_base_regexes
from .synth_pipeline import _render_instruction
from ..fa.automata import ast_to_dfa


def _sample_hard_program(
    inputs: Dict[str, Regex], num_ops: int
) -> tuple[Program, dict]:
    aliases = list(inputs.keys())
    binary_ops = ["UNION", "INTER", "CONCAT"]
    unary_ops = ["STAR", "COMPL", "REVERSE"]

    ops: list[Op] = []
    frags: dict = {}
    have_out = False

    for step in range(num_ops):
        if step == 0:
            if len(aliases) >= 2 and random.random() > 0.3:
                kind = random.choice(binary_ops)
                ops.append(Op(kind=kind, args=[aliases[0], aliases[1]]))
                have_out = True
            else:
                kind = random.choice(unary_ops + ["REPEAT"])
                if kind == "REPEAT":
                    m = random.randint(0, 3)
                    n = random.randint(m, m + 3)
                    ops.append(Op(kind=kind, args=[aliases[0], str(m), str(n)]))
                else:
                    ops.append(Op(kind=kind, args=[aliases[0]]))
                have_out = True
        else:
            if have_out and random.random() > 0.4:
                kind = random.choice(binary_ops)
                other = random.choice(aliases)
                ops.append(Op(kind=kind, args=["out", other]))
            else:
                kind = random.choice(unary_ops + ["REPEAT"])
                target = "out" if have_out else aliases[0]
                if kind == "REPEAT":
                    m = random.randint(0, 3)
                    n = random.randint(m, m + 3)
                    ops.append(Op(kind=kind, args=[target, str(m), str(n)]))
                else:
                    ops.append(Op(kind=kind, args=[target]))
            have_out = True

    prog = Program(inputs=aliases, ops=ops)
    return prog, frags


def _verify_equivalence_timed(
    result_ast: Regex, alphabet: Set[str], timeout_seconds: float
) -> bool:
    start = time.time()
    try:
        ast_to_dfa(result_ast, alphabet)
        return (time.time() - start) < timeout_seconds
    except Exception:
        return False


def generate_hard_sample(
    seed: int,
    alphabet: Set[str] | None = None,
    timeout_seconds: float = 30.0,
    target_num_ops: int | None = None,
) -> Optional[Dict]:
    random.seed(seed)
    if alphabet is None:
        alphabet = DEFAULT_ALPHABET

    if target_num_ops is None:
        target_num_ops = random.choice([4, 5, 6, 7])

    bases = sample_base_regexes(k=2, max_depth=3, alphabet=alphabet)
    inputs: Dict[str, Regex] = {"r1": bases[0], "r2": bases[1]}

    prog, frags = _sample_hard_program(inputs, target_num_ops)

    unary_only = all(
        op.kind in ("STAR", "COMPL", "REVERSE", "REPEAT") for op in prog.ops
    )
    if unary_only:
        inputs = {"r1": inputs["r1"]}
        prog = Program(inputs=["r1"], ops=prog.ops, output_alias=prog.output_alias)

    try:
        result_ast = apply_ops(inputs, prog, alphabet, allow_extended=True)
    except Exception:
        return None

    if not _verify_equivalence_timed(result_ast, alphabet, timeout_seconds):
        return None

    record = {
        "inputs": {k: to_str(v, allow_extended=True) for k, v in inputs.items()},
        "instruction": _render_instruction(prog.ops, frags, inputs),
        "ops_dsl": str(prog),
        "gold_regex": to_str(result_ast, allow_extended=True),
        "meta": {
            "ops": [op.kind for op in prog.ops],
            "seed": seed,
            "frags": frags,
            "num_ops": len(prog.ops),
            "ambiguity_tags": [],
        },
    }
    return record


def generate_hard_corpus(
    n: int,
    path: str,
    alphabet: Set[str] | None = None,
    timeout_seconds: float = 30.0,
) -> Dict:
    per_op = n // 4
    remainder = n - per_op * 4
    targets = {4: per_op, 5: per_op, 6: per_op, 7: per_op + remainder}

    stats = {
        "total_attempted": 0,
        "total_generated": 0,
        "total_filtered": 0,
        "per_op_generated": {k: 0 for k in [4, 5, 6, 7]},
        "per_op_filtered": {k: 0 for k in [4, 5, 6, 7]},
    }

    seed_counter = 0
    with open(path, "w", encoding="utf-8") as f:
        for num_ops, target_count in targets.items():
            generated = 0
            while generated < target_count:
                stats["total_attempted"] += 1
                rec = generate_hard_sample(
                    seed=seed_counter,
                    alphabet=alphabet,
                    timeout_seconds=timeout_seconds,
                    target_num_ops=num_ops,
                )
                seed_counter += 1
                if rec is not None:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    generated += 1
                    stats["total_generated"] += 1
                    stats["per_op_generated"][num_ops] += 1
                else:
                    stats["total_filtered"] += 1
                    stats["per_op_filtered"][num_ops] += 1

                if stats["total_attempted"] > n * 10:
                    break

    return stats
