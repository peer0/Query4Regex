from query4regex.regex.ast import Sym, UnionR, InterR, Star, Concat, Compl
from query4regex.fa.automata import ast_to_dfa
from query4regex.fa.equivalence import dfa_equivalent
from query4regex.config import DEFAULT_ALPHABET as A

def test_union_and_intersection_smoke():
    r = UnionR(Sym("a"), Sym("b"))
    i = InterR(Sym("a"), Sym("b"))
    da = ast_to_dfa(r, A)
    di = ast_to_dfa(i, A)
    assert not da.is_empty()
    assert di.is_empty()  # a ∩ b over {a,b} is empty for single symbols

def test_complement_smoke():
    c = Compl(Sym("a"))
    dc = ast_to_dfa(c, A)
    assert not dc.is_empty()

def test_star_smoke():
    s = Star(Sym("a"))
    ds = ast_to_dfa(s, A)
    assert not ds.is_empty()
