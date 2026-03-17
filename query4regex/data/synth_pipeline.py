from __future__ import annotations
import json, random
from pathlib import Path
import yaml
from typing import Dict, Set
from ..regex.ast import Regex, Concat, UnionR, InterR, Star, Compl, Reverse
from ..regex.pretty import to_str
from ..config import DEFAULT_ALPHABET, DEFAULT_LANG
from ..ops.apply_ops import apply_ops
from ..data.synth_spec import sample_base_regexes
from ..ops.op_dsl import Program, Op
from ..regex.fragments import enumerate_fragments, enumerate_operators

# Load templates for the default language (English)
_TEMPL_PATH = Path(__file__).resolve().parent.parent / 'nl' / f'templates_{DEFAULT_LANG}.yaml'
_TEMPL = yaml.safe_load(open(_TEMPL_PATH, 'r', encoding='utf-8'))

def _render_instruction(ops: list[Op], frags: dict, inputs: dict) -> str:
    parts: list[str] = []

    def _disp_alias(a: str) -> str:
        return "the result" if a =="out" else a

    for op in ops:
        if op.kind == 'REPLACE_OPERAND':
            tmpl = random.choice(_TEMPL['REPLACE_OPERAND'])
            r1_alias = op.args[0]
            idx_str = op.args[1]
            repl_alias = op.args[2] if len(op.args) > 2 else None
            if idx_str in frags:
                frag_text = frags[idx_str]
            else:
                try:
                    idx = int(idx_str)
                    frag_list = enumerate_fragments(inputs[r1_alias])
                    frag_node = frag_list[idx] if 0 <= idx < len(frag_list) else None
                    frag_text = to_str(frag_node) if frag_node is not None else f"fragment #{idx_str}"
                except Exception:
                    frag_text = f"fragment #{idx_str}"
            if repl_alias is not None and repl_alias in inputs:
                r2_text = repl_alias
            else:
                r2_text = _disp_alias(repl_alias) if repl_alias is not None else "<missing>"
            parts.append(tmpl.format(r1=r1_alias, r2=r2_text, frag1=frag_text))
        elif op.kind == 'REPEAT':
            tmpl = random.choice(_TEMPL[op.kind])
            min_rep = op.args[1]
            max_rep = op.args[2] if len(op.args) > 2 else min_rep
            parts.append(tmpl.format(r1=_disp_alias(op.args[0]), min_rep=min_rep, max_rep=max_rep))
        elif op.kind == 'REPLACE_OPERATOR':
            tmpl = random.choice(_TEMPL['REPLACE_OPERATOR'])
            r1_alias, idx_str, new_op = op.args
            old_op = idx_str # Fallback
            try:
                idx = int(idx_str)
                r1_ast = inputs[r1_alias]
                op_nodes = enumerate_operators(r1_ast)
                if 0 <= idx < len(op_nodes):
                    node = op_nodes[idx]
                    if isinstance(node, Concat): old_op = "CONCAT"
                    elif isinstance(node, UnionR): old_op = "UNION"
                    elif isinstance(node, InterR): old_op = "INTER"
                    elif isinstance(node, Star): old_op = "STAR"
                    elif isinstance(node, Compl): old_op = "COMPL"
                    elif isinstance(node, Reverse): old_op = "REVERSE"
            except (ValueError, IndexError, KeyError):
                pass # Use fallback
            parts.append(tmpl.format(r1=_disp_alias(r1_alias), old_op=old_op, new_op=new_op))
        else:
            tmpl = random.choice(_TEMPL[op.kind])
            r1 = op.args[0]
            r2 = _disp_alias(op.args[1]) if len(op.args) > 1 else None
            parts.append(tmpl.format(r1=r1, r2=r2))
    return parts[0] if len(parts) == 1 else ' Then, '.join(parts)

def _sample_program(inputs: Dict[str, Regex]) -> tuple[Program, dict]:
    aliases = list(inputs.keys())
    # Single-step program for now
    op_choices = ['UNION','INTER','COMPL','CONCAT','STAR','REVERSE', 'REPLACE_OPERAND', 'REPLACE_OPERATOR', 'REPEAT']
    kind = random.choice(op_choices)

    # Get the regex we're going to modify for op-specific logic
    alias_to_modify = aliases[0]
    regex_to_modify = inputs[alias_to_modify]

    binary_ops = ['UNION', 'INTER', 'CONCAT']
    unary_ops = ['STAR', 'COMPL', 'REVERSE']

    steps = random.choices([1,2,3], weights=[0.3,0.4,0.3], k=1)[0]
    ops: list[Op] = []
    frags: dict = {}
    have_out = False
    last_bin_aliases: tuple[str, str] | None = None

    modified_aliases: Set[str] = set()

    def add_op(op: Op) -> None:
        nonlocal have_out, last_bin_aliases
        ops.append(op)

        if op.kind in ('REPLACE_OPERAND', 'REPLACE_OPERATOR', 'REPEAT', 'STAR', 'COMPL', 'REVERSE'):
            modified_aliases.add(op.args[0])

        if op.kind in binary_ops:
            have_out = True
            last_bin_aliases = (op.args[0], op.args[1])
        else:
            last_bin_aliases  = last_bin_aliases

    # Fallback for binary ops when we only have one regex
    if len(aliases) < 2 and kind in ('UNION', 'INTER', 'CONCAT', 'REPLACE_OPERAND'):
        kind = random.choice(unary_ops + ['REPEAT'])

    if kind in binary_ops:
        add_op(Op(kind=kind, args=[aliases[0], aliases[1]]))

    elif kind == 'REPLACE_OPERATOR':
        binary_op_nodes = [
            (i, node) for i, node in enumerate(enumerate_operators(regex_to_modify))
            if isinstance(node, (Concat, UnionR, InterR))
        ]
        if binary_op_nodes:
            op_idx, node = random.choice(binary_op_nodes)
            
            current_op_kind = ""
            if isinstance(node, Concat): current_op_kind = "CONCAT"
            elif isinstance(node, UnionR): current_op_kind = "UNION"
            elif isinstance(node, InterR): current_op_kind = "INTER"

            available_new_ops = [op for op in binary_ops if op != current_op_kind]
            if not available_new_ops:
                new_op_kind = random.choice(binary_ops)
            else:
                new_op_kind = random.choice(available_new_ops)

            add_op(Op(kind=kind, args=[alias_to_modify, str(op_idx), new_op_kind]))
            have_out = True
        else:
            add_op(Op(kind=random.choice(unary_ops), args=[alias_to_modify]))

    elif kind == 'REPLACE_OPERAND':
        fragments = enumerate_fragments(regex_to_modify)
        if fragments and len(aliases) > 1:
            i = random.choice(range(len(fragments)))
            frags[str(i)] = to_str(fragments[i])
            add_op(Op(kind=kind, args=[alias_to_modify, str(i), aliases[1]]))
            have_out = True
        else:
            add_op(Op(kind=random.choice(unary_ops), args=[alias_to_modify]))
    
    elif kind in binary_ops:
        args = [aliases[0], aliases[1]]

    elif kind == 'REPEAT':
        min_rep = random.randint(0, 5)
        max_rep = random.randint(min_rep, min_rep + 5) if random.random() > 0.5 else None
        args = [alias_to_modify, str(min_rep)] + ([str(max_rep)] if max_rep is not None else [])
        add_op(Op(kind=kind, args=args))
    else:
        add_op(Op(kind=kind, args=[alias_to_modify]))

    for step in range(1, steps):
        follow_choices = []

        if have_out and len(aliases) >= 1:
            follow_choices += binary_ops

        if last_bin_aliases is not None:
            follow_choices += unary_ops + ['REPEAT']

        unmodified_aliases = [a for a in aliases if a not in modified_aliases]
        if unmodified_aliases:
            pass

        if not follow_choices:
            continue

        kind2 = random.choice(follow_choices)

        if kind2 in binary_ops:
            if have_out:
                other = random.choice(aliases)
                add_op(Op(kind=kind2, args=['out', other]))
            else:
                add_op(Op(kind=kind2, args=[aliases[0], aliases[1]]))
        elif kind2 in unary_ops + ['REPEAT']:
            target_candidates: list[str] = []
            if last_bin_aliases is not None:
                target_candidates.extend(list(last_bin_aliases))

            if have_out:
                target_candidates.append('out')

            if not target_candidates:
                target_candidates = [aliases[0]]

            target = 'out' if have_out else random.choice(target_candidates)

            if kind2 == 'REPEAT':
                m = random.randint(0, 5)
                n = random.randint(m, m + 5) if random.random() > 0.5 else None
                args = [target, str(m)] + ([str(n)] if n is not None else [])
                add_op(Op(kind=kind2, args=args))
            else:
                add_op(Op(kind=kind2, args=[target]))
        elif kind2 == 'REPLACE_OPERAND':
            tgt = random.choice(unmodified_aliases)
            base = inputs[tgt]
            fragment = enumerate_fragments(base)
            if fragment and len(aliases) > 1:
                i = random.randrange(len(fragment))
                frags[str(i)] = to_str(fragment[i])
                other_alias = aliases[1] if tgt == aliases[0] else aliases[0]
                add_op(Op(kind=kind2, args=[tgt, str(i), other_alias]))
                have_out = True
                last_bin_aliases = None
            else:
                add_op(Op(kind=random.choice(unary_ops), args=[aliases[0]]))
        else: # REPLACE_OPERATOR
            tgt = random.choice(unmodified_aliases)
            binary_op_nodes = [
                (i, node) for i, node in enumerate(enumerate_operators(inputs[tgt]))
                if isinstance(node, (Concat, UnionR, InterR))
            ]
            if binary_op_nodes:
                op_idx, node = random.choice(binary_op_nodes)
                
                current_op_kind = ""
                if isinstance(node, Concat): current_op_kind = "CONCAT"
                elif isinstance(node, UnionR): current_op_kind = "UNION"
                elif isinstance(node, InterR): current_op_kind = "INTER"

                available_new_ops = [op for op in binary_ops if op != current_op_kind]
                if not available_new_ops:
                    new_op_kind = random.choice(binary_ops)
                else:
                    new_op_kind = random.choice(available_new_ops)

                add_op(Op(kind='REPLACE_OPERATOR', args=[tgt, str(op_idx), new_op_kind]))
                have_out = True
                last_bin_aliases = None
            else:
                add_op(Op(kind=random.choice(unary_ops), args=[aliases[0]]))

    prog =  Program(inputs=aliases, ops=ops)
    return prog, frags

def generate_sample(seed: int, alphabet: Set[str] | None = None, allow_extended: bool = True) -> Dict:
    random.seed(seed)
    if alphabet is None:
        alphabet = DEFAULT_ALPHABET

    # Determine the number of regexes needed for the operation
    bases = sample_base_regexes(k=2, max_depth=3, alphabet=alphabet)
    inputs: Dict[str, Regex] = {'r1': bases[0], 'r2': bases[1]}
    
    prog, frags = _sample_program(inputs)
    unary_only = all(op.kind in ('STAR','COMPL','REVERSE','REPEAT') for op in prog.ops)

    if unary_only:
        inputs = {'r1': inputs['r1']}
        prog = Program(inputs=['r1'], ops=prog.ops, output_alias=prog.output_alias)

    result_ast = apply_ops(inputs, prog, alphabet, allow_extended=allow_extended)

    record = {
        'inputs': {k: to_str(v, allow_extended=allow_extended) for k,v in inputs.items()},
        'instruction': _render_instruction(prog.ops, frags, inputs),
        'ops_dsl': str(prog),
        'gold_regex': to_str(result_ast, allow_extended=allow_extended),
        'meta': {'ops': [op.kind for op in prog.ops], 'seed': seed, 'frags': frags}
    }
    return record

def generate_corpus(n: int, path: str, alphabet: Set[str] | None = None, allow_extended: bool = True) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(n):
            rec = generate_sample(seed=i, alphabet=alphabet, allow_extended=allow_extended)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
