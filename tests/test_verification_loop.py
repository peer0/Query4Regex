from query4regex.verify.loop import VerificationResult, run_verification_loop
from query4regex.verify.feedback import FeedbackLevel
from query4regex.config import DEFAULT_ALPHABET


class MockModel:
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
    model = MockModel(["\\boxed{a)}", "\\boxed{a|b}"])
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
