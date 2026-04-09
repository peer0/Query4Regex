from query4regex.verify.counterexample import find_counterexample
from query4regex.config import DEFAULT_ALPHABET


def test_counterexample_for_nonequivalent_regexes():
    ce = find_counterexample("a|b", "a", DEFAULT_ALPHABET)
    assert ce is not None
    assert ce.string == "b"
    assert ce.accepted_by == "gold"
    assert ce.rejected_by == "predicted"


def test_counterexample_for_equivalent_regexes():
    ce = find_counterexample("a|b", "b|a", DEFAULT_ALPHABET)
    assert ce is None


def test_counterexample_star_vs_single():
    ce = find_counterexample("a*", "a", DEFAULT_ALPHABET)
    assert ce is not None
    assert ce.string in ("", "aa", "aaa")


def test_counterexample_concat_vs_union():
    """(a)(b) vs a|b — should find counterexample like 'a' or 'ab'."""
    ce = find_counterexample("(a)(b)", "a|b", DEFAULT_ALPHABET)
    assert ce is not None


def test_counterexample_direction_predicted_accepts():
    ce = find_counterexample("a", "a|b", DEFAULT_ALPHABET)
    assert ce is not None
    assert ce.string == "b"
    assert ce.accepted_by == "predicted"
    assert ce.rejected_by == "gold"
