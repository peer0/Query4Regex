from query4regex.constrained.grammar import get_regex_grammar_str
from query4regex.regex.parse import parse_regex_basic
from query4regex.config import DEFAULT_ALPHABET


def test_grammar_string_is_valid():
    grammar = get_regex_grammar_str(DEFAULT_ALPHABET)
    assert isinstance(grammar, str)
    assert len(grammar) > 0
    assert "regex" in grammar.lower() or "union" in grammar.lower()


def test_grammar_examples_are_parsable():
    examples = ["a", "a|b", "(a|b)*", "a(b)*", "(a|b){2,5}", "a*"]
    for ex in examples:
        ast = parse_regex_basic(ex)
        assert ast is not None, f"Failed to parse: {ex}"
