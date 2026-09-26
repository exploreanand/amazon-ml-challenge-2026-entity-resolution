"""How two business names differ, used to limit the cross-encoder veto (v8.1).

On US/India validation, pairs XGBoost is sure about (p >= 0.995) whose names
differ only in spelling, in the legal form ("SCI" / "S.C.I.", an added
"SASU"), or by an added family/group suffix ("& Sons", "Group") are real
matches in 99.95%+ of cases, and the cross-encoder's rare "no" on them is
right less than half the time. Its "no" is reliable (82%) only when the core
names differ, so only those pairs may be vetoed.
"""

import re

from .features import LEGAL

# family / group / partner suffixes that variants of a name add, in the languages seen in the data
_SUFFIX = re.compile(r"\b(sons|associates|holdings?|group|partners|development"
                     r"|fils|associes|groupe|developpement)\b")
KINDS = ("same_name", "legal_form_only", "suffix_added", "other")


def _tokens(text):
    """Tokens with runs of single letters joined, so the dotted "s c i" becomes "sci"."""
    out, run = [], ""
    for w in text.split():
        if len(w) == 1:
            run += w
            continue
        if run:
            out.append(run)
            run = ""
        out.append(w)
    if run:
        out.append(run)
    return out


def variant_kind(b1, b2):
    """b1, b2: ``basic``-normalised names -> one of KINDS."""
    if b1.replace(" ", "") == b2.replace(" ", ""):
        return "same_name"
    if sorted(t for t in _tokens(b1) if t not in LEGAL) == sorted(t for t in _tokens(b2) if t not in LEGAL):
        return "legal_form_only"
    if _SUFFIX.search(b2) and not _SUFFIX.search(b1):
        return "suffix_added"
    return "other"


def adds_country(b1, b2, country):
    """True if the S2/S3 name b2 carries the country word that the S1 name b1 lacks ("Toac Pharmacie" / "Toac France").

    All arguments are ``basic``-normalised. On India validation not one of 84,907
    such candidates is a true match: variants keep an S1's "(India)" but never add it.
    """
    c = f" {country} "
    return c in f" {b2} " and c not in f" {b1} "
