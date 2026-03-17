# Extended Regex Supported

- Union `|`, concatenation (implicit), Kleene star `*`, plus `+`, optional `?`, parentheses `(...)`.
- Extended ops are kept at AST level: intersection `&`, complement `~(...)`, reversal `rev(...)`.
- Special atoms: `ε` (epsilon), `∅` (empty language), `a..z` literals.
