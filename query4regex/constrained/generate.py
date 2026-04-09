from __future__ import annotations
from typing import Set, Any
from .grammar import get_regex_grammar_str


def create_constrained_generator(
    model: Any,
    tokenizer: Any,
    alphabet: Set[str],
) -> Any:
    import outlines

    grammar_str = get_regex_grammar_str(alphabet)
    generator = outlines.generate.cfg(model, grammar_str)
    return generator
