import re
from typing import Dict
from ..config import DEFAULT_LANG

# English-first canonical tokens
CANON_MAP = {
    " union ": [" union "],
    " intersection ": [" intersection "],
    " complement ": [" complement ", " not ", " exclude ", " excluding "],
    " concat ": [" concat ", " concatenate ", " followed by "],
    " star ": [" kleene star ", " zero or more ", " repetition ", " repeat ", " star "],
    " reverse ": [" reverse ", " reversal ", " reversed "],
}

def _normalise_english(s: str) -> str:
    s = " " + s.lower().strip() + " "
    s = re.sub(r"[^a-z0-9_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    for canon, variants in CANON_MAP.items():
        for v in variants:
            s = s.replace(v, canon)
    return s.strip()

def normalize(s: str) -> str:
    # For now only English is implemented; hook for future 'ko'
    return _normalise_english(s)
