from __future__ import annotations
from dataclasses import dataclass
from typing import List, Literal, Optional

OpKind = Literal['UNION','INTER','COMPL','CONCAT','STAR','REVERSE','REPLACE_OPERAND','REPLACE_OPERATOR', 'REPEAT']

@dataclass(frozen=True)
class Op:
    kind: OpKind
    args: List[str]

@dataclass(frozen=True)
class Program:
    inputs: List[str]
    ops: List[Op]
    output_alias: Optional[str] = 'out'

    def __str__(self) -> str:
        ops_str = ' ; '.join(f"{o.kind}({','.join(o.args)})" for o in self.ops)
        return f"inputs={self.inputs} :: {ops_str} -> {self.output_alias}"
