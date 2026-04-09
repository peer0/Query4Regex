from __future__ import annotations
from typing import Set


def get_regex_grammar_str(alphabet: Set[str]) -> str:
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
