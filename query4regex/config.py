from typing import Set, Iterable

# Default global alphabet
DEFAULT_ALPHABET: Set[str] = {"a", "b"}

# Default language for natural-language instructions
DEFAULT_LANG: str = "en"

def make_alphabet(it: Iterable[str]) -> Set[str]:
    res: Set[str] = set()
    for s in it:
        if s:
            res.add(s[0])
    return res
