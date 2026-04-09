import json
import tempfile
import os
from query4regex.data.synth_hard import generate_hard_sample, generate_hard_corpus
from query4regex.config import DEFAULT_ALPHABET


def test_hard_sample_has_4_to_7_ops():
    for seed in range(20):
        rec = generate_hard_sample(seed=seed, alphabet=DEFAULT_ALPHABET)
        if rec is None:
            continue
        n_ops = len(rec["meta"]["ops"])
        assert 4 <= n_ops <= 7, f"seed={seed} has {n_ops} ops"


def test_hard_sample_has_required_fields():
    rec = generate_hard_sample(seed=0, alphabet=DEFAULT_ALPHABET)
    if rec is None:
        return
    required = {"inputs", "instruction", "ops_dsl", "gold_regex", "meta"}
    assert required.issubset(rec.keys())
    assert "ambiguity_tags" in rec["meta"]


def test_hard_corpus_generation():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        stats = generate_hard_corpus(
            n=20, path=path, alphabet=DEFAULT_ALPHABET, timeout_seconds=30
        )
        assert os.path.exists(path)
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) <= 20
        assert stats["total_attempted"] == stats["total_generated"] + stats["total_filtered"]
    finally:
        os.unlink(path)
