# Closing the NL-DSL Gap: Constrained Decoding and Iterative Verification for Regex Transformation

**Date:** 2026-04-09
**Branch:** `extension/constrained-verify-scale`
**Status:** Design approved, pending implementation plan
**Venue target:** Full conference paper (8+ pages)

---

## 1. Thesis

The EACL 2026 Query4Regex paper established that formal DSL queries outperform natural language queries by 6.74 percentage points for regex transformation tasks. This extension investigates whether that gap is caused by **resolvable ambiguity** rather than fundamental reasoning limitations.

**Core claim:** Grammar-constrained decoding eliminates syntax noise, and iterative verification with formal feedback lets models resolve NL ambiguity through self-correction — closing the NL-DSL gap even as compositional complexity increases.

**Implication:** Formal specification (DSL) is unnecessary when formal verification is available. Users can write natural language and achieve equivalent results.

---

## 2. Three Technical Contributions

### 2.1 Grammar-Constrained Decoding

Force models to only generate syntactically valid regexes by integrating a CFG constraint into the decoding process.

**Grammar** derived from the recursive descent parser in `query4regex/regex/parse.py`:

```
regex   -> union
union   -> concat ('|' concat)*
concat  -> postfix+
postfix -> atom ('*' | '+' | '?' | '{n}' | '{n,m}' | '{n,}')*
atom    -> symbol | 'epsilon' | 'empty' | '(' regex ')'
symbol  -> 'a' | 'b' | ...  (from alphabet)
```

**Library:** Outlines (primary), transformers-cfg (fallback).

**Integration:** New `--constrained` flag on `scripts/run_inference.py`. The `\boxed{<valid_regex>}` extraction format is included in the grammar so the model generates the boxing wrapper with a valid regex inside.

**Purpose:** Eliminates all syntactic errors (which account for 539/1000 failures in the best EACL setting), so every verification round is spent on semantic self-correction rather than wasted on unparsable outputs.

### 2.2 Iterative Verification Loop

A multi-turn generate-verify-feedback loop where the model gets formal feedback on incorrect attempts and can self-correct.

**Loop structure:**

```
Input: source regexes, query, model, feedback_type, max_rounds=5

For round k = 1 to max_rounds:
    if k == 1:
        prompt = base_prompt(inputs, query)
    else:
        prompt = base_prompt + conversation_history + feedback[k-1]

    response = model.generate(prompt)
    predicted_regex = extract_regex(response)

    if not parsable(predicted_regex):
        feedback[k] = syntax_feedback(predicted_regex)
        continue

    equivalent, details = verify(predicted_regex, gold_regex)
    if equivalent:
        return Success(round=k)

    feedback[k] = construct_feedback(feedback_type, details)

return Failure(history)
```

**Three feedback granularity levels (ablation conditions):**

**Level A — Binary:**
> "Your answer is incorrect. Try again."

**Level B — Counterexample:**
Extract an accepting path from the symmetric-difference DFA — a concrete string that distinguishes the two languages.
> "Your regex `(a|b)*` is incorrect. For example, the string `aab` is accepted by your regex but should be rejected by the target."

Or the reverse direction:
> "The string `ab` should be accepted by the target regex but your regex `a*` rejects it."

**Level C — Diagnostic:**
Counterexample string + intermediate step verification. Run the gold DSL program step-by-step via `apply_ops()`, compare each intermediate result against what the model's output implies.
> "Your regex `(a|b)*` is incorrect. The string `aab` should be rejected. The expected result after step 1 (CONCAT r1, r2) is `(a)(b)`, and after step 2 (COMPL) is `~((a)(b))`. Your output diverges at step 2 — it appears you applied STAR instead of COMPL."

**Diagnostic step-divergence detection (Level C):**
1. Compute gold intermediate regexes at each step from `apply_ops()`
2. Check if the model's final output is equivalent to any intermediate result (partial correctness — model stopped early)
3. Check if it's equivalent to applying a different operation at some step (operation confusion)

**Verification budget:** Fixed at k=5 rounds. Results reported at each k=1,2,3,4,5 to show the self-correction trajectory. The budget parameter is configurable for future experiments.

**Prompt format:** Multi-turn conversation appending the model's previous attempt and feedback each round. Uses the model's context window naturally.

### 2.3 Query4Regex-Hard: Scaled Compositional Complexity

A new benchmark extending the original 1,000 instances to longer operation chains.

| Parameter | Original | Query4Regex-Hard |
|---|---|---|
| Operations per instance | 1-3 | 4-7 |
| Alphabet | {a, b} | {a, b} |
| Base regex max depth | 3 | 3 |
| Total instances | 1,000 | 1,000 |
| Distribution | 67/18/15% for 1/2/3 ops | 25% each for 4/5/6/7 ops |

**Design rationale:**
- Alphabet and base regex depth are kept identical to isolate compositional complexity as the only independent variable. "Base regex depth" refers to the complexity of the *input* seed regexes (r1, r2) before any operations are applied. The final output regex depth grows naturally with the operation chain length.
- Uniform distribution (250 instances per operation count) for equal statistical power.
- Same 6 operation types as the original (UNION, INTER, CONCAT, COMPL, STAR, REPEAT). No new operations — avoids confounding chain length with novel operations.

**Adaptive timeout filtering:** Discard instances where DFA equivalence checking takes >30 seconds. Log discard rates per operation count as data about formal verification scalability.

**Few-shot pool:** Separate `train_hard.jsonl` with 4-7 ops examples, disjoint from test set.

**NL ambiguity tagging:** Each instance is automatically tagged with applicable ambiguity types at generation time (since we control the NL templates).

---

## 3. NL Ambiguity Taxonomy

Systematic classification of where and why NL queries cause failures that DSL queries avoid.

| Ambiguity Type | Description | NL Example | Why DSL Avoids It |
|---|---|---|---|
| **Scope ambiguity** | Unclear which sub-expression an operation applies to | "Concatenate r1 and r2, then star the result" — does "the result" mean the concat output or r2? | Explicit variable binding: `CONCAT(r1,r2) -> o1; STAR(o1) -> out` |
| **Anaphoric reference** | Pronouns with ambiguous antecedents | "Take r1, complement it, then intersect it with r2" — second "it" = complement result or r1? | Named intermediate variables eliminate pronoun ambiguity |
| **Operator scope** | Unclear nesting of operations | "Union r1 and r2 then repeat 3 to 5 times" — repeat the union, or repeat r2? | `UNION(r1,r2) -> o1; REPEAT(o1,3,5) -> out` — explicit nesting |
| **Implicit ordering** | Order of operations not specified | "Apply star and complement to r1" — which first? Order changes the result | Sequential ordering is explicit in DSL |
| **Fragment reference** | Vague sub-expression identification | "Replace the second part of r1 with r2" — "second part" depends on parse tree traversal | `REPLACE_OPERAND(r1, idx=2, r2)` — explicit index |

**Analysis methodology:**
1. Tag each benchmark instance with applicable ambiguity types (automated from NL templates)
2. Measure per-type accuracy: NL vs DSL breakdown by ambiguity type
3. Measure per-type verification benefit: how much does each feedback level close the gap per ambiguity type?
4. Verification trajectory by type: at which round (k=1..5) does each type converge?

**Key analysis questions:**
- Which ambiguity types are **resolved** by counterexample feedback (B)?
- Which are **only resolved** by diagnostic feedback (C)?
- Which are **unresolvable** even with full diagnostic feedback? (fundamental NL limitation)

---

## 4. Experimental Design

### 4.1 Models

Same 6 models as the EACL paper:
- Non-reasoning: Phi-4 (14B), gemma-3 (27B), Llama-3.3 (70B)
- Reasoning: Phi-4-reasoning (14B), gpt-oss (20B), gpt-oss (120B)

All local HuggingFace models with 4-bit NF4 quantization via BitsAndBytes. Greedy decoding for reproducibility. Constrained decoding is applied as an additional grammar mask on top of greedy decoding — it restricts the token space at each step but does not change the decoding strategy.

### 4.2 Experimental Matrix

| Condition | Query Type | Constrained Decoding | Verification | Feedback Level | Rounds |
|---|---|---|---|---|---|
| Baseline (EACL) | qNL, qDSL | No | No | — | 1 |
| +Constrained | qNL, qDSL | Yes | No | — | 1 |
| +Verify (A) | qNL, qDSL | Yes | Yes | Binary | 1-5 |
| +Verify (B) | qNL, qDSL | Yes | Yes | Counterexample | 1-5 |
| +Verify (C) | qNL, qDSL | Yes | Yes | Diagnostic | 1-5 |

Each condition is run on:
- Original benchmark (1K instances, 1-3 ops)
- Query4Regex-Hard (1K instances, 4-7 ops)

### 4.3 Evaluation Metrics

**From EACL (retained):**
- Syntactic Correctness (Syn.) — always 100% when constrained decoding is on
- Semantic Equivalence (Sem.)
- Sem.dagger — semantic equivalence on syntactically valid outputs only

**New metrics:**
- Accuracy@k — semantic equivalence at verification round k (k=1..5)
- NL-DSL Gap@k — `Sem(DSL, k) - Sem(NL, k)` at each round, tracking gap closure
- Self-correction rate — fraction of initially wrong answers corrected by round k
- Per-ambiguity-type accuracy — breakdown by the 5 ambiguity categories
- Verification convergence speed — average round at which correct answer is reached, by ambiguity type

### 4.4 Key Comparisons

1. **Primary (equal conditions):** NL+constrained+verify vs DSL+constrained+verify — does the gap close?
2. **Practical:** NL+constrained+verify vs DSL single-shot — do users need DSL at all?
3. **Feedback ablation:** Binary (A) vs Counterexample (B) vs Diagnostic (C) — which feedback level is necessary?
4. **Scaling:** Do findings 1-3 hold on Query4Regex-Hard (4-7 ops)?

### 4.5 Success Criteria

- **Strong success:** NL+constrained+verify(C) matches DSL+constrained+verify(C) within 2%p on original benchmark
- **Moderate success:** NL+constrained+verify(C) beats DSL single-shot
- **Scaling success:** The NL-DSL gap does not widen on Query4Regex-Hard compared to original
- **Analysis success:** At least 3/5 ambiguity types show measurable gap reduction from verification

---

## 5. Paper Structure

1. **Introduction** — EACL found NL < DSL. We ask: is this gap fundamental or resolvable?
2. **Background & Query4Regex recap**
3. **Method**
   - 3.1 Grammar-constrained decoding for regex generation
   - 3.2 Iterative verification with formal feedback (3 granularity levels)
   - 3.3 Query4Regex-Hard: scaling compositional complexity to 4-7 ops
4. **NL Ambiguity Taxonomy** — 5 ambiguity types, tagging methodology
5. **Experiments**
   - 5.1 Does constrained decoding close the syntax gap?
   - 5.2 Does verification feedback close the semantic gap? (feedback ablation)
   - 5.3 Does NL + full pipeline match/beat DSL? (primary research question)
   - 5.4 Does the gap stay closed under harder composition? (Query4Regex-Hard)
6. **Analysis**
   - 6.1 Which ambiguity types does verification resolve?
   - 6.2 Verification trajectory: convergence speed by ambiguity type
   - 6.3 Characterizing the irreducible NL-DSL gap
7. **Conclusion**

---

## 6. Implementation Approach

**Order (Verification-First):**
1. Multi-turn verification loop infrastructure
2. Counterexample extraction from symmetric-difference DFA
3. Diagnostic feedback with intermediate step verification
4. Grammar-constrained decoding integration (Outlines)
5. Query4Regex-Hard benchmark generation
6. NL ambiguity auto-tagging
7. Full experimental runs
8. Analysis scripts

**Codebase integration:** All new code extends the existing Query4Regex codebase. New modules are added alongside existing ones; existing code is not modified unless necessary for integration points (e.g., adding flags to `run_inference.py`).

---

## 7. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| NL-DSL gap doesn't close | Paper thesis weakened | Still publishable: "verification helps but gap persists" + analysis of why is a valid finding |
| Constrained decoding too slow | Experiments take too long | Use smaller grammar; benchmark on subset first; try transformers-cfg as lighter alternative |
| DFA equivalence timeout on Hard benchmark | Missing data points at high op counts | Adaptive timeout filtering; report discard rates as data |
| Counterexample extraction fails for some instances | Incomplete feedback in some rounds | Fall back to binary feedback; report fallback rate |
| Diagnostic step-divergence detection is unreliable | Level C feedback is noisy | Conservative matching: only report divergence when confident; otherwise fall back to Level B |
