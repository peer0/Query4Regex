from __future__ import annotations
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class Epsilon: ...
@dataclass(frozen=True)
class Empty: ...
@dataclass(frozen=True)
class Sym: sym: str

@dataclass(frozen=True)
class Concat: left: 'Regex'; right: 'Regex'
@dataclass(frozen=True)
class UnionR: left: 'Regex'; right: 'Regex'
@dataclass(frozen=True)
class InterR: left: 'Regex'; right: 'Regex'
@dataclass(frozen=True)
class Star: inner: 'Regex'
@dataclass(frozen=True)
class Compl: inner: 'Regex'
@dataclass(frozen=True)
class Reverse: inner: 'Regex'
@dataclass(frozen=True)
class Repeat: inner: 'Regex'; min: int; max: int | None

Regex = Union[Epsilon, Empty, Sym, Concat, UnionR, InterR, Star, Compl, Reverse, Repeat]
