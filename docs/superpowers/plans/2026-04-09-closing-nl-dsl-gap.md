# Closing the NL-DSL Gap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build infrastructure for grammar-constrained decoding, multi-turn iterative verification with three feedback levels, and a scaled benchmark (4-7 ops) — to test whether NL + verification can close the performance gap with DSL.

**Architecture:** New modules are added alongside the existing `query4regex/` package. The verification loop (`query4regex/verify/`) handles multi-turn feedback construction and counterexample extraction. Constrained decoding (`query4regex/constrained/`) wraps the Outlines library to enforce regex grammar during generation. The scaled benchmark extends the existing `synth_pipeline.py` with a new `generate_hard_corpus()` entry point. A new `scripts/run_inference_verified.py` orchestrates the full pipeline.

**Tech Stack:** Python 3.11, pyformlang (DFA ops), Outlines (constrained decoding), transformers + BitsAndBytes (inference)

---

## File Map

**New files to create:**

| File | Responsibility |
|---|---|
| `query4regex/verify/__init__.py` | Package init |
| `query4regex/verify/counterexample.py` | Extract distinguishing strings from symmetric-difference DFA |
| `query4regex/verify/intermediate.py` | Compute gold intermediate regexes per DSL step; detect step divergence |
| `query4regex/verify/feedback.py` | Construct feedback strings at levels A/B/C |
| `query4regex/verify/loop.py` | Multi-turn verification loop orchestrator |
| `query4regex/constrained/__init__.py` | Package init |
| `query4regex/constrained/grammar.py` | Regex CFG definition for Outlines |
| `query4regex/constrained/generate.py` | Wrap model generation with grammar constraint |
| `query4regex/data/synth_hard.py` | Generate Query4Regex-Hard (4-7 ops) with timeout filtering |
| `query4regex/data/ambiguity_tagger.py` | Auto-tag NL instances with ambiguity types |
| `scripts/run_inference_verified.py` | Main experiment runner with verification loop + constrained decoding |
| `scripts/analyze_ambiguity.py` | Per-ambiguity-type accuracy and convergence analysis |
| `tests/test_counterexample.py` | Tests for counterexample extraction |
| `tests/test_intermediate.py` | Tests for intermediate step verification |
| `tests/test_feedback.py` | Tests for feedback construction |
| `tests/test_verification_loop.py` | Tests for the full verification loop |
| `tests/test_constrained_grammar.py` | Tests for grammar-constrained decoding |
| `tests/test_synth_hard.py` | Tests for hard benchmark generation |
| `tests/test_ambiguity_tagger.py` | Tests for ambiguity tagging |

**Existing files to modify:**

| File | Change |
|---|---|
| `requirements.txt` | Add `outlines`, `timeout-decorator` (if missing) |

---

### Task 1: Counterexample Extraction from Symmetric-Difference DFA

**Files:**
- Create: `query4regex/verify/__init__.py`
- Create: `query4regex/verify/counterexample.py`
- Test: `tests/test_counterexample.py`

This is the foundation for feedback levels B and C. Given two non-equivalent regexes, extract a concrete string accepted by one but not the other.

- [ ] **Step 1: Write the failing test**

Create `tests/test_counterexample.py`:

```python
from query4regex.verify.counterexample import find_counterexample
from query4regex.config import DEFAULT_ALPHABET


def test_counterexample_for_nonequivalent_regexes():
    """a|b vs a — should find 'b' as counterexample."""
    ce = find_counterexample("a|b", "a", DEFAULT_ALPHABET)
    assert ce is not None
    assert ce.string == "b"
    assert ce.accepted_by == "gold"
    assert ce.rejected_by == "predicted"


def test_counterexample_for_equivalent_regexes():
    """a|b vs b|a — equivalent, no counterexample."""
    ce = find_counterexample("a|b", "b|a", DEFAULT_ALPHABET)
    assert ce is None


def test_counterexample_star_vs_single():
    """a* vs a — should find counterexample like '' or 'aa'."""
    ce = find_counterexample("a*", "a", DEFAULT_ALPHABET)
    assert ce is not None
    assert ce.string in ("", "aa", "aaa")


def test_counterexample_complement():
    """~(a) vs a — very different languages, should find counterexample."""
    ce = find_counterexample("~(a)", "a", DEFAULT_ALPHABET)
    assert ce is not None


def test_counterexample_direction_predicted_accepts():
    """a vs a|b — predicted accepts 'b' but gold doesn't."""
    ce = find_counterexample("a", "a|b", DEFAULT_ALPHABET)
    assert ce is not None
    assert ce.string == "b"
    assert ce.accepted_by == "predicted"
    assert ce.rejected_by == "gold"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_counterexample.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'query4regex.verify'`

- [ ] **Step 3: Create package init**

Create `query4regex/verify/__init__.py`:

```python
```

- [ ] **Step 4: Implement counterexample extraction**

Create `query4regex/verify/counterexample.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Set, Optional
from pyformlang.finite_automaton import DeterministicFiniteAutomaton, State, Symbol
from ..eval.custom_equivalence import regex_to_nfa


@dataclass(frozen=True)
class Counterexample:
    string: str
    accepted_by: str   # "gold" or "predicted"
    rejected_by: str   # "gold" or "predicted"


def _dfa_from_regex_str(regex_str: str) -> DeterministicFiniteAutomaton:
    """Convert a regex string (with extended ops) to a minimized DFA."""
    nfa = regex_to_nfa(regex_str)
    return nfa.to_deterministic().minimize()


def _find_accepting_path(dfa: DeterministicFiniteAutomaton) -> Optional[str]:
    """BFS through DFA to find shortest accepted string. Returns None if language is empty."""
    if dfa.is_empty():
        return None
    start_states = list(dfa.start_states)
    if not start_states:
        return None

    from collections import deque
    queue = deque()
    visited = set()

    for s in start_states:
        queue.append((s, ""))
        visited.add(s)

    while queue:
        state, path = queue.popleft()
        if state in dfa.final_states:
            return path
        transitions = dfa.to_dict()
        for (src, sym), dsts in transitions.items():
            if src == state and sym is not None:
                for dst in dsts:
                    if dst not in visited:
                        visited.add(dst)
                        queue.append((dst, path + sym.value))
    return None


def find_counterexample(
    gold_regex: str,
    predicted_regex: str,
    alphabet: Set[str],
) -> Optional[Counterexample]:
    """
    Find a distinguishing string between two regexes.

    Returns a Counterexample with the string and which regex accepts/rejects it,
    or None if the regexes are equivalent.
    """
    try:
        gold_dfa = _dfa_from_regex_str(gold_regex)
        pred_dfa = _dfa_from_regex_str(predicted_regex)
    except Exception:
        return None

    # gold accepts but predicted doesn't: gold ∩ ~predicted
    try:
        pred_compl = pred_dfa.get_complement(
            alphabet={Symbol(a) for a in alphabet}
        )
        gold_minus_pred = gold_dfa.get_intersection(pred_compl).minimize()
        s = _find_accepting_path(gold_minus_pred)
        if s is not None:
            return Counterexample(
                string=s, accepted_by="gold", rejected_by="predicted"
            )
    except Exception:
        pass

    # predicted accepts but gold doesn't: predicted ∩ ~gold
    try:
        gold_compl = gold_dfa.get_complement(
            alphabet={Symbol(a) for a in alphabet}
        )
        pred_minus_gold = pred_dfa.get_intersection(gold_compl).minimize()
        s = _find_accepting_path(pred_minus_gold)
        if s is not None:
            return Counterexample(
                string=s, accepted_by="predicted", rejected_by="gold"
            )
    except Exception:
        pass

    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_counterexample.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add query4regex/verify/__init__.py query4regex/verify/counterexample.py tests/test_counterexample.py
git commit -m "feat: add counterexample extraction from symmetric-difference DFA"
```

---

### Task 2: Intermediate Step Verification

**Files:**
- Create: `query4regex/verify/intermediate.py`
- Test: `tests/test_intermediate.py`

Compute gold intermediate regex at each DSL program step, then detect where a model's output diverges.

- [ ] **Step 1: Write the failing test**

Create `tests/test_intermediate.py`:

```python
from query4regex.verify.intermediate import (
    compute_gold_intermediates,
    detect_divergence,
)
from query4regex.ops.op_dsl import Program, Op
from query4regex.regex.ast import Sym, UnionR
from query4regex.regex.parse import parse_regex_basic
from query4regex.config import DEFAULT_ALPHABET


def test_compute_intermediates_two_steps():
    """UNION(r1,r2) -> o1; STAR(o1) -> out should produce 2 intermediates."""
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
    """Model output matches step 0 (UNION) but not step 1 (STAR) — stopped early."""
    inputs = {"r1": Sym("a"), "r2": Sym("b")}
    prog = Program(
        inputs=["r1", "r2"],
        ops=[
            Op(kind="UNION", args=["r1", "r2"]),
            Op(kind="STAR", args=["out"]),
        ],
    )
    intermediates = compute_gold_intermediates(inputs, prog, DEFAULT_ALPHABET)
    # Model output is a|b (matches step 0 but not step 1 which should be (a|b)*)
    divergence = detect_divergence("a|b", intermediates, DEFAULT_ALPHABET)
    assert divergence is not None
    assert divergence.matches_step == 0
    assert divergence.diverges_at_step == 1
    assert divergence.diagnosis == "partial_correctness"


def test_detect_divergence_fully_correct():
    """Model output matches final step — no divergence."""
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_intermediate.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement intermediate step verification**

Create `query4regex/verify/intermediate.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from ..regex.ast import Regex
from ..regex.pretty import to_str
from ..ops.op_dsl import Program
from ..ops.apply_ops import apply_ops as _full_apply_ops
from ..fa.automata import ast_to_dfa
from ..fa.equivalence import dfa_equivalent
from ..eval.custom_equivalence import regex_to_nfa


@dataclass(frozen=True)
class StepResult:
    step_index: int
    op_kind: str
    regex_str: str
    regex_ast: Regex


@dataclass(frozen=True)
class Divergence:
    matches_step: int          # last step the output matches (-1 if matches none)
    diverges_at_step: int      # first step the output doesn't match
    diagnosis: str             # "partial_correctness" | "operation_confusion" | "unknown"
    description: str           # human-readable explanation


def compute_gold_intermediates(
    inputs: Dict[str, Regex],
    program: Program,
    alphabet: Set[str],
) -> List[StepResult]:
    """Execute the DSL program step-by-step, recording each intermediate result."""
    from ..ops.op_dsl import Program as Prog, Op
    results = []

    for i in range(len(program.ops)):
        partial_prog = Prog(
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


def _are_equivalent_safe(
    regex_str: str, step: StepResult, alphabet: Set[str]
) -> bool:
    """Check equivalence between a regex string and a step result, with error handling."""
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
    """
    Check if predicted regex matches any intermediate step.

    Returns None if it matches the final step (fully correct).
    Returns a Divergence describing where the model went wrong.
    """
    if not intermediates:
        return None

    # Check if it matches the final step (correct)
    final = intermediates[-1]
    if _are_equivalent_safe(predicted_regex, final, alphabet):
        return None

    # Find the latest step that the prediction matches
    matches_step = -1
    for step in intermediates[:-1]:
        if _are_equivalent_safe(predicted_regex, step, alphabet):
            matches_step = step.step_index

    diverges_at = matches_step + 1

    if matches_step >= 0:
        expected_op = intermediates[diverges_at].op_kind
        return Divergence(
            matches_step=matches_step,
            diverges_at_step=diverges_at,
            diagnosis="partial_correctness",
            description=(
                f"Your output matches the result after step {matches_step + 1} "
                f"({intermediates[matches_step].op_kind}), but diverges at step "
                f"{diverges_at + 1} ({expected_op})."
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_intermediate.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add query4regex/verify/intermediate.py tests/test_intermediate.py
git commit -m "feat: add intermediate step verification for diagnostic feedback"
```

---

### Task 3: Feedback Construction (Levels A, B, C)

**Files:**
- Create: `query4regex/verify/feedback.py`
- Test: `tests/test_feedback.py`

Build the three feedback levels using counterexample extraction and intermediate verification.

- [ ] **Step 1: Write the failing test**

Create `tests/test_feedback.py`:

```python
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
    assert "b" in fb  # the counterexample string
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
    assert "step" in fb.lower()  # mentions which step diverged


def test_counterexample_fallback_on_failure():
    """If counterexample extraction fails, fall back to binary."""
    fb = construct_feedback(
        level=FeedbackLevel.COUNTEREXAMPLE,
        gold_regex="a|b",
        predicted_regex="a",
        alphabet=DEFAULT_ALPHABET,
    )
    # Should still produce some feedback even if implementation detail varies
    assert len(fb) > 0


def test_syntax_feedback():
    fb = construct_feedback(
        level=FeedbackLevel.BINARY,
        gold_regex="a|b",
        predicted_regex="",
        alphabet=DEFAULT_ALPHABET,
        is_unparsable=True,
    )
    assert "parsable" in fb.lower() or "valid" in fb.lower() or "syntax" in fb.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feedback.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement feedback construction**

Create `query4regex/verify/feedback.py`:

```python
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
    """
    Construct feedback at the given level.

    Level A (BINARY): "Your answer is incorrect. Try again."
    Level B (COUNTEREXAMPLE): Binary + a concrete distinguishing string.
    Level C (DIAGNOSTIC): Counterexample + which intermediate step diverged.

    Falls back to a simpler level if the richer feedback cannot be computed.
    """
    if is_unparsable:
        return _SYNTAX_MSG

    if level == FeedbackLevel.BINARY:
        return _BINARY_MSG

    # Level B and C both need a counterexample
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_feedback.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add query4regex/verify/feedback.py tests/test_feedback.py
git commit -m "feat: add feedback construction at three granularity levels (A/B/C)"
```

---

### Task 4: Verification Loop Orchestrator

**Files:**
- Create: `query4regex/verify/loop.py`
- Test: `tests/test_verification_loop.py`

The core multi-turn loop: generate → verify → feedback → repeat.

- [ ] **Step 1: Write the failing test**

Create `tests/test_verification_loop.py`:

```python
from query4regex.verify.loop import (
    VerificationResult,
    run_verification_loop,
)
from query4regex.verify.feedback import FeedbackLevel
from query4regex.config import DEFAULT_ALPHABET


class MockModel:
    """A mock model that returns a sequence of pre-defined responses."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._call_count = 0

    def generate(self, prompt: str) -> str:
        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1
        return self._responses[idx]

    @property
    def call_count(self) -> int:
        return self._call_count


def test_correct_on_first_attempt():
    model = MockModel(["\\boxed{a|b}"])
    result = run_verification_loop(
        model=model,
        gold_regex="a|b",
        inputs={"r1": "a", "r2": "b"},
        instruction="Take the union of r1 and r2.",
        ops_dsl="inputs=['r1','r2'] :: UNION(r1,r2) -> out",
        pipeline="nl2regex",
        alphabet=DEFAULT_ALPHABET,
        feedback_level=FeedbackLevel.BINARY,
        max_rounds=5,
    )
    assert result.success is True
    assert result.solved_at_round == 1
    assert model.call_count == 1


def test_correct_on_second_attempt():
    model = MockModel(["\\boxed{a}", "\\boxed{a|b}"])
    result = run_verification_loop(
        model=model,
        gold_regex="a|b",
        inputs={"r1": "a", "r2": "b"},
        instruction="Take the union of r1 and r2.",
        ops_dsl="inputs=['r1','r2'] :: UNION(r1,r2) -> out",
        pipeline="nl2regex",
        alphabet=DEFAULT_ALPHABET,
        feedback_level=FeedbackLevel.COUNTEREXAMPLE,
        max_rounds=5,
    )
    assert result.success is True
    assert result.solved_at_round == 2
    assert model.call_count == 2


def test_never_correct():
    model = MockModel(["\\boxed{a}"] * 5)
    result = run_verification_loop(
        model=model,
        gold_regex="a|b",
        inputs={"r1": "a", "r2": "b"},
        instruction="Take the union of r1 and r2.",
        ops_dsl="inputs=['r1','r2'] :: UNION(r1,r2) -> out",
        pipeline="nl2regex",
        alphabet=DEFAULT_ALPHABET,
        feedback_level=FeedbackLevel.BINARY,
        max_rounds=3,
    )
    assert result.success is False
    assert result.solved_at_round is None
    assert len(result.rounds) == 3


def test_unparsable_then_correct():
    model = MockModel(["\\boxed{(((}", "\\boxed{a|b}"])
    result = run_verification_loop(
        model=model,
        gold_regex="a|b",
        inputs={"r1": "a", "r2": "b"},
        instruction="Take the union of r1 and r2.",
        ops_dsl="inputs=['r1','r2'] :: UNION(r1,r2) -> out",
        pipeline="nl2regex",
        alphabet=DEFAULT_ALPHABET,
        feedback_level=FeedbackLevel.COUNTEREXAMPLE,
        max_rounds=5,
    )
    assert result.success is True
    assert result.solved_at_round == 2
    assert result.rounds[0].parsable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_verification_loop.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement the verification loop**

Create `query4regex/verify/loop.py`:

```python
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set
from ..config import DEFAULT_ALPHABET
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
    """Extract regex from \\boxed{...} format."""
    match = re.search(r'oxed\{(.*)\}', answer, re.DOTALL)
    return match.group(1) if match else ""


def _is_parsable(regex_str: str) -> bool:
    """Check if a regex string can be parsed."""
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
    """Build the prompt including conversation history for multi-turn."""
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

    # Append conversation history
    parts = [base]
    for rec in history:
        parts.append(f" \\boxed{{{rec.predicted_regex}}}")
        parts.append(f"\n\nFeedback: {rec.feedback}\n\nRevised regex:")
    return "".join(parts)


def _parse_program_from_dsl(ops_dsl: str) -> Optional[Program]:
    """Parse a DSL string back into a Program. Returns None on failure."""
    try:
        # Format: inputs=['r1','r2'] :: UNION(r1,r2) -> o1 ; STAR(o1) -> out
        header, body = ops_dsl.split("::", 1)
        # Extract input names
        import ast as stdlib_ast
        inputs_str = header.strip().replace("inputs=", "")
        input_names = stdlib_ast.literal_eval(inputs_str)

        steps = [s.strip() for s in body.strip().split(";")]
        ops = []
        for step in steps:
            # e.g. "UNION(r1,r2) -> o1"
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
    """
    Run the multi-turn verification loop.

    Args:
        model: Any object with a .generate(prompt) -> str method.
        gold_regex: The correct target regex string.
        inputs: Dict of input regex names to regex strings (e.g. {"r1": "a", "r2": "b"}).
        instruction: The NL instruction string.
        ops_dsl: The DSL program string.
        pipeline: "nl2regex" or "nl2dsl2regex".
        alphabet: Set of alphabet symbols.
        feedback_level: Which feedback granularity to use.
        max_rounds: Maximum verification attempts.
        input_asts: Pre-parsed input ASTs (for diagnostic feedback). Auto-parsed if None.

    Returns:
        VerificationResult with success status and per-round records.
    """
    history: List[RoundRecord] = []

    # Parse inputs for diagnostic feedback
    parsed_inputs: Optional[Dict[str, Regex]] = input_asts
    parsed_program: Optional[Program] = None
    if feedback_level == FeedbackLevel.DIAGNOSTIC:
        if parsed_inputs is None:
            try:
                parsed_inputs = {
                    k: parse_regex_basic(v) for k, v in inputs.items()
                }
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_verification_loop.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add query4regex/verify/loop.py tests/test_verification_loop.py
git commit -m "feat: add multi-turn verification loop orchestrator"
```

---

### Task 5: Grammar-Constrained Decoding via Outlines

**Files:**
- Create: `query4regex/constrained/__init__.py`
- Create: `query4regex/constrained/grammar.py`
- Create: `query4regex/constrained/generate.py`
- Modify: `requirements.txt` (add `outlines`)
- Test: `tests/test_constrained_grammar.py`

- [ ] **Step 1: Add outlines to requirements**

Append to `requirements.txt`:

```
outlines
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_constrained_grammar.py`:

```python
from query4regex.constrained.grammar import get_regex_grammar_str
from query4regex.regex.parse import parse_regex_basic
from query4regex.config import DEFAULT_ALPHABET


def test_grammar_string_is_valid():
    """Grammar string should be a non-empty string."""
    grammar = get_regex_grammar_str(DEFAULT_ALPHABET)
    assert isinstance(grammar, str)
    assert len(grammar) > 0
    assert "regex" in grammar.lower() or "union" in grammar.lower()


def test_grammar_examples_are_parsable():
    """Regexes that match the grammar should be parsable by our parser."""
    examples = ["a", "a|b", "(a|b)*", "a(b)*", "(a|b){2,5}", "a*"]
    for ex in examples:
        ast = parse_regex_basic(ex)
        assert ast is not None, f"Failed to parse: {ex}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_constrained_grammar.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Implement the grammar module**

Create `query4regex/constrained/__init__.py`:

```python
```

Create `query4regex/constrained/grammar.py`:

```python
from __future__ import annotations
from typing import Set


def get_regex_grammar_str(alphabet: Set[str]) -> str:
    """
    Return an EBNF grammar string for valid regexes over the given alphabet.
    Compatible with Outlines' CFG-based constrained generation.

    The grammar mirrors the parser in query4regex/regex/parse.py:
      regex   -> union
      union   -> concat ('|' concat)*
      concat  -> postfix+
      postfix -> atom ('*' | '+' | '?' | '{' digits ',' digits '}')*
      atom    -> symbol | '(' regex ')'
      symbol  -> 'a' | 'b' | ...
    """
    symbols = " | ".join(f'"{s}"' for s in sorted(alphabet))

    grammar = f"""
?start: boxed

boxed: "\\\\boxed{{" regex "}}"

?regex: union

?union: concat ("|" concat)*

?concat: postfix+

?postfix: atom quantifier*

?quantifier: "*" | "+" | "?" | "{{" DIGITS "," DIGITS "}}" | "{{" DIGITS "}}"

?atom: symbol | "(" regex ")"

?symbol: {symbols}

DIGITS: /[0-9]+/
"""
    return grammar.strip()
```

Create `query4regex/constrained/generate.py`:

```python
from __future__ import annotations
from typing import Set, Any
from .grammar import get_regex_grammar_str


def create_constrained_generator(
    model: Any,
    tokenizer: Any,
    alphabet: Set[str],
) -> Any:
    """
    Wrap a HuggingFace model with Outlines grammar-constrained generation.

    Args:
        model: A loaded HuggingFace model.
        tokenizer: The corresponding tokenizer.
        alphabet: Set of alphabet symbols for the regex grammar.

    Returns:
        An Outlines generator that only produces syntactically valid regexes.
    """
    import outlines

    grammar_str = get_regex_grammar_str(alphabet)
    generator = outlines.generate.cfg(model, grammar_str)
    return generator
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_constrained_grammar.py -v`
Expected: All 2 tests PASS

- [ ] **Step 6: Commit**

```bash
git add query4regex/constrained/__init__.py query4regex/constrained/grammar.py query4regex/constrained/generate.py tests/test_constrained_grammar.py requirements.txt
git commit -m "feat: add grammar-constrained decoding via Outlines"
```

---

### Task 6: Query4Regex-Hard Benchmark Generation

**Files:**
- Create: `query4regex/data/synth_hard.py`
- Test: `tests/test_synth_hard.py`

Generate the scaled benchmark with 4-7 operations and adaptive timeout filtering.

- [ ] **Step 1: Write the failing test**

Create `tests/test_synth_hard.py`:

```python
import json
import tempfile
import os
from query4regex.data.synth_hard import generate_hard_sample, generate_hard_corpus
from query4regex.config import DEFAULT_ALPHABET


def test_hard_sample_has_4_to_7_ops():
    """Each hard sample should have between 4 and 7 operations."""
    for seed in range(20):
        rec = generate_hard_sample(seed=seed, alphabet=DEFAULT_ALPHABET)
        if rec is None:
            continue  # filtered by timeout
        n_ops = len(rec["meta"]["ops"])
        assert 4 <= n_ops <= 7, f"seed={seed} has {n_ops} ops"


def test_hard_sample_has_required_fields():
    """Hard samples should have the same fields as original samples."""
    rec = generate_hard_sample(seed=0, alphabet=DEFAULT_ALPHABET)
    if rec is None:
        return  # timeout-filtered
    required = {"inputs", "instruction", "ops_dsl", "gold_regex", "meta"}
    assert required.issubset(rec.keys())
    assert "ambiguity_tags" in rec["meta"]


def test_hard_corpus_generation():
    """Generate a small hard corpus and verify structure."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        stats = generate_hard_corpus(
            n=20, path=path, alphabet=DEFAULT_ALPHABET, timeout_seconds=30
        )
        assert os.path.exists(path)
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) <= 20
        assert stats["total_attempted"] == 20
        assert stats["total_generated"] == len(lines)
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_synth_hard.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement hard benchmark generation**

Create `query4regex/data/synth_hard.py`:

```python
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
from ..fa.equivalence import dfa_equivalent
from ..regex.fragments import enumerate_fragments, enumerate_operators


def _sample_hard_program(
    inputs: Dict[str, Regex], num_ops: int
) -> tuple[Program, dict]:
    """Sample a program with exactly num_ops operations (4-7)."""
    aliases = list(inputs.keys())
    binary_ops = ["UNION", "INTER", "CONCAT"]
    unary_ops = ["STAR", "COMPL", "REVERSE"]
    all_ops = binary_ops + unary_ops + ["REPEAT"]

    ops: list[Op] = []
    frags: dict = {}
    have_out = False

    for step in range(num_ops):
        if step == 0:
            # First op: use original inputs
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
            # Subsequent ops: operate on 'out' or combine with inputs
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
    """Check that the generated regex is verifiable within the timeout."""
    start = time.time()
    try:
        dfa = ast_to_dfa(result_ast, alphabet)
        elapsed = time.time() - start
        return elapsed < timeout_seconds
    except Exception:
        return False


def generate_hard_sample(
    seed: int,
    alphabet: Set[str] | None = None,
    timeout_seconds: float = 30.0,
    target_num_ops: int | None = None,
) -> Optional[Dict]:
    """
    Generate a single hard benchmark instance with 4-7 operations.

    Returns None if the instance is filtered out by timeout.
    """
    random.seed(seed)
    if alphabet is None:
        alphabet = DEFAULT_ALPHABET

    if target_num_ops is None:
        target_num_ops = random.choice([4, 5, 6, 7])

    bases = sample_base_regexes(k=2, max_depth=3, alphabet=alphabet)
    inputs: Dict[str, Regex] = {"r1": bases[0], "r2": bases[1]}

    prog, frags = _sample_hard_program(inputs, target_num_ops)

    # Check if only unary ops — reduce to single input
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

    # Check verifiability within timeout
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
            "ambiguity_tags": [],  # filled by tagger in Task 7
        },
    }
    return record


def generate_hard_corpus(
    n: int,
    path: str,
    alphabet: Set[str] | None = None,
    timeout_seconds: float = 30.0,
) -> Dict:
    """
    Generate a hard corpus with uniform distribution over 4-7 ops.

    Returns statistics about generation (attempted, generated, filtered, per-op counts).
    """
    per_op = n // 4  # 250 each for 4,5,6,7
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

                # Safety: don't loop forever
                if stats["total_attempted"] > n * 10:
                    break

    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_synth_hard.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add query4regex/data/synth_hard.py tests/test_synth_hard.py
git commit -m "feat: add Query4Regex-Hard benchmark generation (4-7 ops)"
```

---

### Task 7: NL Ambiguity Auto-Tagger

**Files:**
- Create: `query4regex/data/ambiguity_tagger.py`
- Test: `tests/test_ambiguity_tagger.py`

Automatically tag each benchmark instance with applicable NL ambiguity types.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ambiguity_tagger.py`:

```python
from query4regex.data.ambiguity_tagger import tag_ambiguity


def test_scope_ambiguity_detected():
    """Multi-step NL with 'the result' should tag scope ambiguity."""
    instruction = "Concatenate r1 and r2. Then, Apply Kleene star to the result."
    ops = ["CONCAT", "STAR"]
    tags = tag_ambiguity(instruction, ops)
    assert "scope" in tags


def test_anaphoric_reference_detected():
    """NL with 'it' referring to previous result should tag anaphoric."""
    instruction = "Take the complement of r1. Then, Compute the intersection of it and r2."
    ops = ["COMPL", "INTER"]
    tags = tag_ambiguity(instruction, ops)
    assert "anaphoric" in tags


def test_implicit_ordering_detected():
    """Multiple ops mentioned without clear ordering."""
    instruction = "Apply star and complement to r1."
    ops = ["STAR", "COMPL"]
    tags = tag_ambiguity(instruction, ops)
    assert "implicit_ordering" in tags


def test_single_op_no_ambiguity():
    """Single unambiguous instruction should have minimal tags."""
    instruction = "Take the union of r1 and r2."
    ops = ["UNION"]
    tags = tag_ambiguity(instruction, ops)
    # Single-op instructions may still have fragment ambiguity but not scope/anaphoric
    assert "scope" not in tags
    assert "anaphoric" not in tags


def test_fragment_reference_detected():
    """Fragment-based operations should tag fragment reference ambiguity."""
    instruction = "In r1, replace the second part with r2."
    ops = ["REPLACE_OPERAND"]
    tags = tag_ambiguity(instruction, ops)
    assert "fragment" in tags
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ambiguity_tagger.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement ambiguity tagger**

Create `query4regex/data/ambiguity_tagger.py`:

```python
from __future__ import annotations
import re
from typing import List


def tag_ambiguity(instruction: str, ops: List[str]) -> List[str]:
    """
    Tag an NL instruction with applicable ambiguity types.

    Ambiguity types:
    - "scope": multi-step with "the result" / "the whole result" (unclear target)
    - "anaphoric": pronouns like "it" / "this" referring to previous output
    - "operator_scope": operator applied after binary op without clear parenthesization
    - "implicit_ordering": multiple ops mentioned without explicit order
    - "fragment": vague fragment references like "the second part", "fragment"

    Returns a list of ambiguity type strings.
    """
    tags: List[str] = []
    lower = instruction.lower()
    is_multi_step = len(ops) > 1

    # Scope ambiguity: multi-step with "the result" / "whole result"
    if is_multi_step and re.search(r'\b(the result|the whole result|the entire result)\b', lower):
        tags.append("scope")

    # Anaphoric reference: pronouns "it" / "this" / "that" in multi-step
    if is_multi_step and re.search(r'\b(it|this|that)\b', lower):
        # Exclude "it" in phrases like "concatenate it" at start (not anaphoric)
        # Only count if "it" appears after a conjunction like "then" or comma
        if re.search(r'(then|,)\s.*\b(it|this|that)\b', lower):
            tags.append("anaphoric")

    # Operator scope: unary op after binary op in same sentence without parens
    if is_multi_step:
        unary_after_binary = False
        binary_ops = {"UNION", "INTER", "CONCAT"}
        unary_ops = {"STAR", "COMPL", "REVERSE", "REPEAT"}
        for i in range(1, len(ops)):
            if ops[i] in unary_ops and ops[i - 1] in binary_ops:
                unary_after_binary = True
        if unary_after_binary and "then" not in lower:
            tags.append("operator_scope")

    # Implicit ordering: "and" connecting two operations without "then"
    op_keywords = ["union", "intersect", "complement", "concatenat", "star", "repeat", "reverse"]
    found_ops = [kw for kw in op_keywords if kw in lower]
    if len(found_ops) >= 2 and " and " in lower and "then" not in lower:
        tags.append("implicit_ordering")

    # Fragment reference: vague sub-expression references
    if re.search(r'\b(part|portion|fragment|section|element)\b', lower):
        tags.append("fragment")
    if any(op in ("REPLACE_OPERAND", "REPLACE_OPERATOR") for op in ops):
        if "fragment" not in tags:
            tags.append("fragment")

    return tags
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ambiguity_tagger.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add query4regex/data/ambiguity_tagger.py tests/test_ambiguity_tagger.py
git commit -m "feat: add NL ambiguity auto-tagger for 5 ambiguity types"
```

---

### Task 8: Experiment Runner with Verification + Constrained Decoding

**Files:**
- Create: `scripts/run_inference_verified.py`

This is the main experiment entry point combining all components. No unit test file — integration tested by running with `--end-idx 5`.

- [ ] **Step 1: Create the experiment runner**

Create `scripts/run_inference_verified.py`:

```python
#!/usr/bin/env python3
"""
Experiment runner for the NL-DSL gap extension.
Combines: constrained decoding + iterative verification + multi-feedback levels.

Usage:
    python scripts/run_inference_verified.py \
        --model-name "microsoft/Phi-4" \
        --dataset-path "./data/test.jsonl" \
        --pipeline "nl2regex" \
        --shot "zero" \
        --feedback-level "counterexample" \
        --max-rounds 5 \
        --constrained
"""
import argparse
import json
import os
import random
import re
import sys
from copy import deepcopy
from string import Template

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query4regex.config import DEFAULT_ALPHABET
from query4regex.verify.feedback import FeedbackLevel
from query4regex.verify.loop import VerificationResult, run_verification_loop


class HFModel:
    """Wraps a HuggingFace model to match the GenerativeModel protocol."""

    def __init__(self, model, tokenizer, generation_config, reasoning=False,
                 thinking_generation_config=None, think_end_token="</think>",
                 constrained_generator=None):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.reasoning = reasoning
        self.thinking_generation_config = thinking_generation_config
        self.think_end_token = think_end_token
        self.constrained_generator = constrained_generator

    def generate(self, prompt: str) -> str:
        input_template = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

        if self.constrained_generator is not None:
            # Use Outlines constrained generation
            result = self.constrained_generator(input_template)
            return result

        inputs = self.tokenizer(
            input_template, add_special_tokens=False, return_tensors="pt"
        ).to("cuda")
        query_length = len(inputs["input_ids"][0])

        if self.reasoning and self.thinking_generation_config is not None:
            think_cfg = self.thinking_generation_config
            if len(self.tokenizer.encode(self.think_end_token, add_special_tokens=False)) == 1:
                output_ids = self.model.generate(**inputs, generation_config=think_cfg)[0]
            else:
                output_ids = self.model.generate(
                    **inputs, generation_config=think_cfg, tokenizer=self.tokenizer
                )[0]
            output_string = self.tokenizer.decode(output_ids[query_length:])
            full_input = input_template + output_string.split(self.think_end_token)[0] + self.think_end_token
            inputs = self.tokenizer(
                full_input, add_special_tokens=False, return_tensors="pt"
            ).to("cuda")

        output_ids = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            tokenizer=self.tokenizer,
        )[0]
        return self.tokenizer.decode(output_ids[query_length:])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, type=str)
    parser.add_argument("--checkpoint-path", default=None, type=str)
    parser.add_argument("--reasoning", action="store_true", default=False)
    parser.add_argument("--dataset-path", default="./data/test.jsonl", type=str)
    parser.add_argument("--fewshot-path", default="./data/train.jsonl", type=str)
    parser.add_argument("--shot", choices=["five", "zero"], default="zero", type=str)
    parser.add_argument("--pipeline", choices=["nl2regex", "nl2dsl2regex"], default="nl2regex", type=str)
    parser.add_argument("--feedback-level", choices=["binary", "counterexample", "diagnostic"], default="counterexample", type=str)
    parser.add_argument("--max-rounds", default=5, type=int)
    parser.add_argument("--constrained", action="store_true", default=False)
    parser.add_argument("--max-think-tokens", default=4096, type=int)
    parser.add_argument("--max-answer-tokens", default=1024, type=int)
    parser.add_argument("--start-idx", default=0, type=int)
    parser.add_argument("--end-idx", default=2147483647, type=int)
    parser.add_argument("--instruction-path", default="query4regex/nl/instruction_template.md", type=str)
    parser.add_argument("--instruction-path-dsl", default="query4regex/nl_dsl/instruction_template.md", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    feedback_map = {
        "binary": FeedbackLevel.BINARY,
        "counterexample": FeedbackLevel.COUNTEREXAMPLE,
        "diagnostic": FeedbackLevel.DIAGNOSTIC,
    }
    feedback_level = feedback_map[args.feedback_level]

    model_name = args.model_name if args.checkpoint_path is None else args.checkpoint_path
    instruction_path = args.instruction_path_dsl if args.pipeline == "nl2dsl2regex" else args.instruction_path

    # Load instruction
    with open(instruction_path, "r") as f:
        instruction_text = f.read()

    # Load dataset
    with open(args.dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Output path
    constrained_tag = "_constrained" if args.constrained else ""
    path = f"./result/verified/{args.pipeline}/{args.shot}-shot/{args.feedback_level}_r{args.max_rounds}{constrained_tag}/"
    os.makedirs(path, exist_ok=True)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", quantization_config=quant_config, device_map="auto"
    )
    model.eval()

    # Generation config
    gen_config = deepcopy(model.generation_config)
    gen_config.max_new_tokens = args.max_answer_tokens
    gen_config.do_sample = False
    gen_config.pad_token_id = tokenizer.eos_token_id

    thinking_gen_config = None
    if args.reasoning:
        thinking_gen_config = deepcopy(model.generation_config)
        thinking_gen_config.max_new_tokens = args.max_think_tokens
        think_end_token = "</think>"
        if len(tokenizer.encode(think_end_token, add_special_tokens=False)) == 1:
            thinking_gen_config.eos_token_id = tokenizer.encode(think_end_token, add_special_tokens=False)[0]
        else:
            thinking_gen_config.stop_strings = think_end_token

    # Constrained decoding setup
    constrained_generator = None
    if args.constrained:
        from query4regex.constrained.generate import create_constrained_generator
        constrained_generator = create_constrained_generator(model, tokenizer, DEFAULT_ALPHABET)

    hf_model = HFModel(
        model=model,
        tokenizer=tokenizer,
        generation_config=gen_config,
        reasoning=args.reasoning,
        thinking_generation_config=thinking_gen_config,
        constrained_generator=constrained_generator,
    )

    # Fewshot prompt
    fewshot_prompt = ""
    if args.shot == "five":
        with open(args.fewshot_path, "r") as f:
            fewshot_data = [json.loads(line) for line in f]
        random.seed(42)
        examples = random.sample(fewshot_data, 5)
        prompt_template = Template(
            "Given the following regular expressions:\n${regex_inputs}\n\n"
            "Instruction: ${instruction}\n\nResulting regex:"
        )
        for ex in examples:
            regex_inputs = "\n".join(f"{k}: {v}" for k, v in ex["inputs"].items())
            fewshot_prompt += prompt_template.substitute(
                regex_inputs=regex_inputs, instruction=ex["instruction"]
            ) + f" {ex['gold_regex']}\n\n"

    # Resume from existing results
    result_path = os.path.join(path, model_name.split("/")[-1] + ".jsonl")
    start_idx = args.start_idx
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            lines = f.readlines()
        if lines:
            try:
                last = json.loads(lines[-1])
                start_idx = max(start_idx, last["idx"] + 1)
            except json.JSONDecodeError:
                pass

    # Run inference with verification
    with open(result_path, "a") as f:
        for i, x in enumerate(tqdm(data[start_idx : args.end_idx])):
            vresult = run_verification_loop(
                model=hf_model,
                gold_regex=x["gold_regex"],
                inputs=x["inputs"],
                instruction=instruction_text + "\n\n" + x["instruction"],
                ops_dsl=x["ops_dsl"],
                pipeline=args.pipeline,
                alphabet=DEFAULT_ALPHABET,
                feedback_level=feedback_level,
                max_rounds=args.max_rounds,
            )

            record = {
                "idx": i + start_idx,
                "inputs": x["inputs"],
                "instruction": x["instruction"],
                "ops_dsl": x["ops_dsl"],
                "gold_regex": x["gold_regex"],
                "success": vresult.success,
                "solved_at_round": vresult.solved_at_round,
                "num_rounds": len(vresult.rounds),
                "rounds": [
                    {
                        "round": r.round_num,
                        "predicted_regex": r.predicted_regex,
                        "parsable": r.parsable,
                        "equivalent": r.equivalent,
                        "feedback": r.feedback,
                    }
                    for r in vresult.rounds
                ],
            }
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses without error**

Run: `python -c "import scripts.run_inference_verified" 2>&1 || python scripts/run_inference_verified.py --help`
Expected: Shows help text without import errors

- [ ] **Step 3: Commit**

```bash
git add scripts/run_inference_verified.py
git commit -m "feat: add experiment runner with verification loop and constrained decoding"
```

---

### Task 9: Ambiguity Analysis Script

**Files:**
- Create: `scripts/analyze_ambiguity.py`

Compute per-ambiguity-type accuracy, NL-DSL gap at each round, and convergence speed.

- [ ] **Step 1: Create analysis script**

Create `scripts/analyze_ambiguity.py`:

```python
#!/usr/bin/env python3
"""
Analyze verification results by ambiguity type.

Reads verified result files and produces:
1. Per-ambiguity-type accuracy at each round k
2. NL-DSL gap at each round k
3. Convergence speed (avg round to correct) per ambiguity type

Usage:
    python scripts/analyze_ambiguity.py \
        --nl-results result/verified/nl2regex/five-shot/counterexample_r5/model.jsonl \
        --dsl-results result/verified/nl2dsl2regex/five-shot/counterexample_r5/model.jsonl \
        --dataset-path data/test.jsonl \
        --output result/analysis/
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query4regex.data.ambiguity_tagger import tag_ambiguity


def load_results(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_dataset(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_accuracy_at_k(results: List[Dict], max_k: int) -> Dict[int, float]:
    """Compute cumulative accuracy at each round k."""
    acc = {}
    for k in range(1, max_k + 1):
        correct = sum(
            1 for r in results
            if r.get("success") and r.get("solved_at_round", max_k + 1) <= k
        )
        acc[k] = correct / max(1, len(results))
    return acc


def analyze(
    nl_results: List[Dict],
    dsl_results: List[Dict],
    dataset: List[Dict],
    max_rounds: int,
) -> Dict:
    """Full analysis: per-type accuracy, gap, convergence."""
    # Tag each instance with ambiguity types
    ambiguity_tags = {}
    for i, item in enumerate(dataset):
        ops = item.get("meta", {}).get("ops", [])
        tags = tag_ambiguity(item.get("instruction", ""), ops)
        ambiguity_tags[i] = tags

    all_types = ["scope", "anaphoric", "operator_scope", "implicit_ordering", "fragment"]

    analysis = {
        "overall": {
            "nl_accuracy_at_k": compute_accuracy_at_k(nl_results, max_rounds),
            "dsl_accuracy_at_k": compute_accuracy_at_k(dsl_results, max_rounds),
        },
        "per_type": {},
        "gap_at_k": {},
        "convergence": {},
    }

    # NL-DSL gap at each k
    for k in range(1, max_rounds + 1):
        nl_acc = analysis["overall"]["nl_accuracy_at_k"][k]
        dsl_acc = analysis["overall"]["dsl_accuracy_at_k"][k]
        analysis["gap_at_k"][k] = dsl_acc - nl_acc

    # Per-type analysis
    for amb_type in all_types:
        # Filter to instances with this ambiguity type
        type_indices = {i for i, tags in ambiguity_tags.items() if amb_type in tags}
        if not type_indices:
            continue

        nl_subset = [r for r in nl_results if r["idx"] in type_indices]
        dsl_subset = [r for r in dsl_results if r["idx"] in type_indices]

        nl_acc = compute_accuracy_at_k(nl_subset, max_rounds)
        dsl_acc = compute_accuracy_at_k(dsl_subset, max_rounds)

        # Convergence speed: average round at which NL gets correct
        nl_rounds = [
            r["solved_at_round"]
            for r in nl_subset
            if r.get("success") and r.get("solved_at_round") is not None
        ]
        avg_convergence = sum(nl_rounds) / max(1, len(nl_rounds)) if nl_rounds else None

        analysis["per_type"][amb_type] = {
            "count": len(type_indices),
            "nl_accuracy_at_k": nl_acc,
            "dsl_accuracy_at_k": dsl_acc,
            "gap_at_k": {k: dsl_acc[k] - nl_acc[k] for k in range(1, max_rounds + 1)},
            "nl_avg_convergence_round": avg_convergence,
        }

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nl-results", required=True, type=str)
    parser.add_argument("--dsl-results", required=True, type=str)
    parser.add_argument("--dataset-path", default="./data/test.jsonl", type=str)
    parser.add_argument("--max-rounds", default=5, type=int)
    parser.add_argument("--output", default="./result/analysis/", type=str)
    args = parser.parse_args()

    nl_results = load_results(args.nl_results)
    dsl_results = load_results(args.dsl_results)
    dataset = load_dataset(args.dataset_path)

    analysis = analyze(nl_results, dsl_results, dataset, args.max_rounds)

    os.makedirs(args.output, exist_ok=True)
    output_path = os.path.join(args.output, "ambiguity_analysis.json")
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print("=== NL-DSL Gap at each round ===")
    for k, gap in analysis["gap_at_k"].items():
        print(f"  k={k}: {gap:+.4f}")

    print("\n=== Per-Ambiguity-Type Summary ===")
    for atype, data in analysis["per_type"].items():
        final_gap = data["gap_at_k"][args.max_rounds]
        conv = data["nl_avg_convergence_round"]
        conv_str = f"{conv:.1f}" if conv is not None else "N/A"
        print(f"  {atype} (n={data['count']}): gap@{args.max_rounds}={final_gap:+.4f}, avg_convergence={conv_str}")

    print(f"\nFull analysis saved to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script parses without error**

Run: `python scripts/analyze_ambiguity.py --help`
Expected: Shows help text

- [ ] **Step 3: Commit**

```bash
git add scripts/analyze_ambiguity.py
git commit -m "feat: add ambiguity analysis script for per-type accuracy and gap tracking"
```

---

### Task 10: Generate Hard Benchmark Data Files

**Files:**
- Uses: `query4regex/data/synth_hard.py` (from Task 6)
- Creates: `data/test_hard.jsonl`, `data/train_hard.jsonl`

- [ ] **Step 1: Create a generation script for the hard benchmark**

Create `scripts/make_hard_dataset.py`:

```python
#!/usr/bin/env python3
"""Generate the Query4Regex-Hard benchmark (4-7 ops)."""
import argparse
import json
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query4regex.config import DEFAULT_ALPHABET
from query4regex.data.synth_hard import generate_hard_corpus
from query4regex.data.ambiguity_tagger import tag_ambiguity


def tag_corpus(path: str) -> None:
    """Add ambiguity tags to each instance in-place."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    for rec in records:
        ops = rec.get("meta", {}).get("ops", [])
        instruction = rec.get("instruction", "")
        rec["meta"]["ambiguity_tags"] = tag_ambiguity(instruction, ops)

    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--out-dir", type=str, default="./data")
    args = parser.parse_args()

    test_path = os.path.join(args.out_dir, "test_hard.jsonl")
    train_path = os.path.join(args.out_dir, "train_hard.jsonl")

    print(f"Generating test set ({args.n_test} instances, 4-7 ops)...")
    test_stats = generate_hard_corpus(args.n_test, test_path, DEFAULT_ALPHABET, args.timeout)
    print(f"  Generated: {test_stats['total_generated']}, Filtered: {test_stats['total_filtered']}")
    print(f"  Per-op: {test_stats['per_op_generated']}")

    print(f"\nGenerating train set ({args.n_train} instances, 4-7 ops)...")
    train_stats = generate_hard_corpus(args.n_train, train_path, DEFAULT_ALPHABET, args.timeout)
    print(f"  Generated: {train_stats['total_generated']}, Filtered: {train_stats['total_filtered']}")

    print("\nTagging ambiguity types...")
    tag_corpus(test_path)
    tag_corpus(train_path)

    print(f"\nDone! Files: {test_path}, {train_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses**

Run: `python scripts/make_hard_dataset.py --help`
Expected: Shows help text

- [ ] **Step 3: Generate the datasets (small test run)**

Run: `python scripts/make_hard_dataset.py --n-test 20 --n-train 20 --out-dir ./data`
Expected: Creates `data/test_hard.jsonl` and `data/train_hard.jsonl` with ~20 instances each

- [ ] **Step 4: Verify data format**

Run: `python -c "import json; data=[json.loads(l) for l in open('data/test_hard.jsonl')]; print(f'count={len(data)}'); print(f'ops_range={min(len(d[\"meta\"][\"ops\"]) for d in data)}-{max(len(d[\"meta\"][\"ops\"]) for d in data)}'); print(f'first={json.dumps(data[0], indent=2)[:200]}')"`
Expected: Shows count, ops range 4-7, and a snippet of the first instance

- [ ] **Step 5: Commit**

```bash
git add scripts/make_hard_dataset.py
git commit -m "feat: add script to generate Query4Regex-Hard benchmark data"
```

---

### Task 11: Tag Ambiguity on Original Dataset

**Files:**
- Uses: `query4regex/data/ambiguity_tagger.py` (from Task 7)

Tag the existing `data/test.jsonl` with ambiguity types for analysis comparisons.

- [ ] **Step 1: Create a tagging script for existing data**

Create `scripts/tag_ambiguity.py`:

```python
#!/usr/bin/env python3
"""Add ambiguity tags to an existing dataset file. Writes to a new file."""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from query4regex.data.ambiguity_tagger import tag_ambiguity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", default=None, type=str, help="Defaults to input path with _tagged suffix")
    args = parser.parse_args()

    output = args.output
    if output is None:
        base, ext = os.path.splitext(args.input)
        output = f"{base}_tagged{ext}"

    records = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    for rec in records:
        ops = rec.get("meta", {}).get("ops", [])
        instruction = rec.get("instruction", "")
        rec["meta"]["ambiguity_tags"] = tag_ambiguity(instruction, ops)

    with open(output, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    all_tags = []
    for rec in records:
        all_tags.extend(rec["meta"]["ambiguity_tags"])
    counts = Counter(all_tags)
    print(f"Tagged {len(records)} instances -> {output}")
    print(f"Tag distribution: {dict(counts)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on original test set**

Run: `python scripts/tag_ambiguity.py --input data/test.jsonl --output data/test_tagged.jsonl`
Expected: Creates tagged file and prints tag distribution

- [ ] **Step 3: Commit**

```bash
git add scripts/tag_ambiguity.py
git commit -m "feat: add script to tag existing datasets with ambiguity types"
```

---

### Task 12: Run Experiments Shell Script

**Files:**
- Create: `run_verified_experiments.sh`

Convenience script to run the full experimental matrix for one model.

- [ ] **Step 1: Create the shell script**

Create `run_verified_experiments.sh`:

```bash
#!/bin/bash
# Run the full experimental matrix for a given model.
# Usage: bash run_verified_experiments.sh <MODEL_NAME> [--reasoning]

MODEL=$1
EXTRA_ARGS="${@:2}"

if [ -z "$MODEL" ]; then
    echo "Usage: bash run_verified_experiments.sh <MODEL_NAME> [--reasoning]"
    exit 1
fi

DATASETS=("./data/test.jsonl" "./data/test_hard.jsonl")
DATASET_NAMES=("original" "hard")
PIPELINES=("nl2regex" "nl2dsl2regex")
SHOTS=("zero" "five")
FEEDBACK_LEVELS=("binary" "counterexample" "diagnostic")

for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    DNAME=${DATASET_NAMES[$i]}

    if [ ! -f "$DATASET" ]; then
        echo "Skipping $DNAME: $DATASET not found"
        continue
    fi

    for PIPELINE in "${PIPELINES[@]}"; do
        for SHOT in "${SHOTS[@]}"; do
            # Baseline: single-shot, no verification, no constrained
            echo "=== $DNAME / $PIPELINE / $SHOT-shot / baseline ==="
            python scripts/run_inference_verified.py \
                --model-name "$MODEL" \
                --dataset-path "$DATASET" \
                --fewshot-path "./data/train.jsonl" \
                --pipeline "$PIPELINE" \
                --shot "$SHOT" \
                --feedback-level "binary" \
                --max-rounds 1 \
                $EXTRA_ARGS

            # Constrained only: single-shot
            echo "=== $DNAME / $PIPELINE / $SHOT-shot / constrained ==="
            python scripts/run_inference_verified.py \
                --model-name "$MODEL" \
                --dataset-path "$DATASET" \
                --fewshot-path "./data/train.jsonl" \
                --pipeline "$PIPELINE" \
                --shot "$SHOT" \
                --feedback-level "binary" \
                --max-rounds 1 \
                --constrained \
                $EXTRA_ARGS

            # Verification loop at each feedback level
            for FEEDBACK in "${FEEDBACK_LEVELS[@]}"; do
                echo "=== $DNAME / $PIPELINE / $SHOT-shot / $FEEDBACK / 5 rounds / constrained ==="
                python scripts/run_inference_verified.py \
                    --model-name "$MODEL" \
                    --dataset-path "$DATASET" \
                    --fewshot-path "./data/train.jsonl" \
                    --pipeline "$PIPELINE" \
                    --shot "$SHOT" \
                    --feedback-level "$FEEDBACK" \
                    --max-rounds 5 \
                    --constrained \
                    $EXTRA_ARGS
            done
        done
    done
done

echo "All experiments complete for $MODEL"
```

- [ ] **Step 2: Make executable and verify**

Run: `chmod +x run_verified_experiments.sh && bash run_verified_experiments.sh 2>&1 | head -3`
Expected: Shows usage message (no model provided)

- [ ] **Step 3: Commit**

```bash
git add run_verified_experiments.sh
git commit -m "feat: add shell script for full experimental matrix"
```
