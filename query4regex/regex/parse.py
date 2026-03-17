# Minimal regex parser for literals, |, concatenation and postfix quantifiers.
from __future__ import annotations
from typing import Tuple
from .ast import *

class Parser:
    def __init__(self, s: str):
        self.s = s
        self.i = 0

    def peek(self) -> str | None:
        return self.s[self.i] if self.i < len(self.s) else None

    def eat(self, ch: str | None = None) -> str:
        c = self.peek()
        if c is None: return ''
        if ch is not None and c != ch:
            raise ValueError(f'Expected {ch} got {c}')
        self.i += 1
        return c

    def parse(self) -> Regex:
        r = self.parse_union()
        if self.peek() is not None:
            raise ValueError('Unexpected trailing input')
        return r

    def parse_union(self) -> Regex:
        left = self.parse_concat()
        while self.peek() == '|':
            self.eat('|')
            right = self.parse_concat()
            left = UnionR(left, right)
        return left

    def parse_concat(self) -> Regex:
        parts = []
        while True:
            c = self.peek()
            if c is None or c in ')|':
                break
            parts.append(self.parse_postfix())
        if not parts: return Epsilon()
        r = parts[0]
        for p in parts[1:]:
            r = Concat(r, p)
        return r

    def parse_postfix(self) -> Regex:
        base = self.parse_atom()
        while True:
            c = self.peek()
            if c == '*':
                self.eat('*'); base = Star(base)
            elif c == '+':
                self.eat('+'); base = Concat(base, Star(base))
            elif c == '?':
                self.eat('?'); base = UnionR(base, Epsilon())
            elif c == '{':
                self.eat('{')
                min_rep_str = ''
                while self.peek().isdigit():
                    min_rep_str += self.eat()
                min_rep = int(min_rep_str)

                max_rep = None
                if self.peek() == ',':
                    self.eat(',')
                    if self.peek().isdigit():
                        max_rep_str = ''
                        while self.peek().isdigit():
                            max_rep_str += self.eat()
                        max_rep = int(max_rep_str)
                
                self.eat('}')
                base = Repeat(base, min_rep, max_rep)
            else:
                break
        return base

    def parse_atom(self) -> Regex:
        c = self.peek()
        if c is None: return Epsilon()
        if c == '(':
            self.eat('(')
            r = self.parse_union()
            self.eat(')')
            return r
        if c == 'ε':
            self.eat('ε'); return Epsilon()
        if c == '∅':
            self.eat('∅'); return Empty()
        # literal
        self.eat()
        return Sym(c)

def parse_regex_basic(s: str) -> Regex:
    return Parser(s).parse()
