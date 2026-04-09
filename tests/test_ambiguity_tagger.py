from query4regex.data.ambiguity_tagger import tag_ambiguity


def test_scope_ambiguity_detected():
    instruction = "Concatenate r1 and r2. Then, Apply Kleene star to the result."
    tags = tag_ambiguity(instruction, ["CONCAT", "STAR"])
    assert "scope" in tags


def test_anaphoric_reference_detected():
    instruction = "Take the complement of r1. Then, Compute the intersection of it and r2."
    tags = tag_ambiguity(instruction, ["COMPL", "INTER"])
    assert "anaphoric" in tags


def test_implicit_ordering_detected():
    instruction = "Apply star and complement to r1."
    tags = tag_ambiguity(instruction, ["STAR", "COMPL"])
    assert "implicit_ordering" in tags


def test_single_op_no_ambiguity():
    instruction = "Take the union of r1 and r2."
    tags = tag_ambiguity(instruction, ["UNION"])
    assert "scope" not in tags
    assert "anaphoric" not in tags


def test_fragment_reference_detected():
    instruction = "In r1, replace the second part with r2."
    tags = tag_ambiguity(instruction, ["REPLACE_OPERAND"])
    assert "fragment" in tags
