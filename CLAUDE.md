# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run experiments (zero-shot + five-shot for a given model)
bash run_experiments.sh <MODEL_NAME>

# Run inference manually
python scripts/run_inference.py \
    --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --dataset-path "./data/test.jsonl" \
    --pipeline "nl2dsl2regex" \
    --shot "zero"

# Five-shot (requires --fewshot-path)
python scripts/run_inference.py \
    --model-name "<MODEL>" \
    --dataset-path "./data/test.jsonl" \
    --fewshot-path "./data/train.jsonl" \
    --pipeline "nl2dsl2regex" \
    --shot "five"

# Partial dataset run
python scripts/run_inference.py --model-name "<MODEL>" \
    --dataset-path "./data/test.jsonl" --start-idx 0 --end-idx 100

# Evaluation
python scripts/calculate_scores.py
python scripts/evaluate_results.py
python scripts/detailed_analyzer.py       # Error analysis by operation type
python scripts/performance_analyzer.py    # Breakdown by complexity

# Dataset generation
python scripts/make_dataset.py --n 1000 --out custom_corpus.jsonl --alphabet a b c

# Tests
python -m tests.test_fa_ops
```

## Architecture

This is a research benchmark evaluating LLM regex transformation using formal verification via DFA equivalence.

### Core Pipeline

```
Source regex(es) + query (qNL or qDSL)
  → LLM generates target regex
  → Parse output (extract from \boxed{...} format)
  → Convert predicted + reference regexes to DFAs (via pyformlang)
  → DFA equivalence check → semantic correctness
```

### Two Query Modes

- **qNL** (natural language): Ambiguous, uses pronouns like "the result" / "it"
- **qDSL** (formal DSL): `inputs=['r1','r2'] :: CONCAT(r1,r2) -> o1; STAR(o1) -> out`

DSL consistently outperforms NL by ~6.74%p across models.

### Two Pipeline Modes

- `nl2regex` — Direct NL → regex generation
- `nl2dsl2regex` — NL → DSL → regex (recommended)

### Key Packages

- **`query4regex/regex/`** — Immutable AST nodes (`Sym`, `Concat`, `UnionR`, `InterR`, `Star`, `Compl`, `Reverse`, `Repeat`). Parser is hand-written recursive descent. `pretty.py` converts AST back to string.
- **`query4regex/fa/`** — DFA construction from AST via pyformlang (`ast_to_dfa`). Equivalence via symmetric difference (`dfa_equivalent`). Simple ops (union/concat/star) go through NFA; complex ops (complement/intersection/reverse) operate directly on DFA objects.
- **`query4regex/ops/`** — `Op` and `Program` dataclasses for DSL representation. `apply_ops.py` executes a Program on regex inputs sequentially.
- **`query4regex/nl/`** — NL → Program parser (heuristic, splits on "then"). YAML templates for NL generation.
- **`query4regex/eval/`** — DFA-based accuracy metrics. Two equivalence implementations: symmetric-difference-based and pyformlang's built-in `is_equivalent_to`. 5-second timeout on equivalence checking.
- **`query4regex/data/`** — Corpus generation: random regex sampling by depth + random operation composition.

### Critical Invariants

- **Alphabet consistency**: All DFA operations must use the same alphabet (default `{"a", "b"}` in `config.py`). Complement and intersection are incorrect without this.
- **Regular language closure**: All supported operations (union, intersection, concat, complement, star, bounded repetition) preserve regularity, ensuring DFA conversion is always possible.
- **Memoization**: `regex_to_nfa()` caches results to avoid recomputation.

### Data Format

Each instance in `test.jsonl` / `train.jsonl`:
```json
{
  "inputs": {"r1": "...", "r2": "..."},
  "instruction": "NL description...",
  "ops_dsl": "inputs=['r1','r2'] :: CONCAT(r1,r2) -> o1; STAR(o1) -> out",
  "gold_regex": "((regex))*",
  "meta": {"ops": ["CONCAT", "STAR"], "seed": 0, "frags": {...}}
}
```

### Result Output

Results go to `result/generation/{pipeline}/{shot}-shot/{model_name}.jsonl` with fields: `gold_regex`, `generated_answer`, `generated_regex`, `idx`, and evaluation metrics.

### Metrics

- **Syn.** — Syntactically valid (parsable regex)?
- **Sem.** — Semantically equivalent (DFA equivalence) over all instances
- **Sem.†** — Semantic equivalence over syntactically valid outputs only

### Debugging Equivalence

```python
from query4regex.eval.custom_equivalence import are_equivalent
are_equivalent("a|b", "b|a")  # True

from query4regex.regex.ast import Sym, UnionR
from query4regex.fa.automata import ast_to_dfa
from query4regex.config import DEFAULT_ALPHABET
dfa = ast_to_dfa(UnionR(Sym("a"), Sym("b")), DEFAULT_ALPHABET)
```

### Environment

- Python 3.11, PyTorch 2.8.0, Transformers 4.56.2
- 4-bit NF4 quantization via BitsAndBytes for inference
- Greedy decoding for reproducibility
- GPU: NVIDIA A6000 or RTX PRO 6000 Blackwell
