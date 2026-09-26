"""French-style copies of training pairs for the cross-encoder (v9).

France is absent from training, and the e5-large cross-encoder misjudges French
name variants that behave exactly like English ones: "SCI" / "S.C.I." (like
"LLC" / "L.L.C."), "& Fils" (like "& Sons"). frenchify rewrites both records
of a training pair the same way, keeping its label, so the model sees those
forms with the labels their English counterparts carry:

  legal forms   LLC, Inc, Ltd, Pvt Ltd, Corp, Co ... -> one French form per pair
                (SARL, SAS, SA, EURL, SASU, SCI, SNC), dotted where the original was
  suffixes      & Sons -> & Fils, Group -> Groupe, Associates -> Associes, ...
  words         and -> et, of -> de, India -> France (names only; addresses are kept)
  type words    (type_words=True) School -> Ecole, Committee -> Comite, Pharmacy -> Pharmacie, ...
                so that a decoy swapping "School" for "College" teaches "Ecole" vs "College"

Only generic words are mapped (no lookup of real businesses or places).
"""

import re

_LEGAL = re.compile(
    r"\b(?:(?:pvt|private|ltd|limited|llc|l\.\s?l\.\s?c|inc|incorporated|corp|corporation|co|company|llp|plc)\.?"
    r"(?:\s+|$))+", re.IGNORECASE)
_FR_FORMS = ("SARL", "SAS", "SA", "EURL", "SASU", "SCI", "SNC")
_WORDS = [(re.compile(rf"\b{a}\b", re.IGNORECASE), b) for a, b in (
    (r"&\s*sons", "& Fils"), (r"and\s+sons", "et Fils"), (r"sons", "Fils"),
    (r"&\s*associates", "& Associés"), (r"associates", "Associés"),
    (r"holdings?", "Holding"), (r"group", "Groupe"), (r"development", "Développement"),
    (r"enterprises?", "Entreprises"), (r"and", "et"), (r"of", "de"), (r"india", "France"),
)]
_TYPES = [(re.compile(rf"\b(?:{a})\b", re.IGNORECASE), b) for a, b in (
    (r"committee", "Comité"), (r"high\s+school", "Lycée"), (r"school", "École"), (r"college", "Collège"),
    (r"academy", "Académie"), (r"pharmacy|chemists?", "Pharmacie"), (r"clinic", "Clinique"), (r"hospital", "Hôpital"),
    (r"health", "Santé"), (r"care", "Soins"), (r"center|centre", "Centre"), (r"house|home", "Maison"),
    (r"sports?", "Sportive"), (r"society", "Société"), (r"foundation", "Fondation"), (r"cultural|culture", "Culture"),
    (r"theatre|theater", "Théâtre"), (r"music", "Musique"), (r"dance", "Danse"), (r"bakery", "Boulangerie"),
    (r"hotel", "Hôtel"), (r"travels?", "Voyages"), (r"market", "Marché"), (r"stores?|shop", "Boutique"),
    (r"motors|automotive|auto", "Automobile"), (r"works|workshop", "Ateliers"), (r"farms?", "Fermes"),
    (r"agro|agricultural", "Agricole"), (r"dental", "Dentaire"), (r"medical", "Médical"), (r"traders|trading", "Commerce"),
    (r"consulting|consultants?", "Conseil"), (r"jewell?ers?", "Bijouterie"), (r"events?", "Fêtes"), (r"leisure", "Loisirs"),
    (r"friends", "Amis"), (r"kindergarten|nursery", "Maternelle"), (r"primary", "Primaire"), (r"elementary", "Élémentaire"),
)]


def _dotted(form):
    return ".".join(form) + "."


def _legal(text, form, rng):
    def sub(m):
        dotted = "." in m.group(0) or rng.random() < 0.15
        tail = " " if m.group(0).endswith(" ") else ""
        return (_dotted(form) if dotted else form) + tail
    return _LEGAL.sub(sub, text)


def frenchify(a, b, rng, type_words=False):
    """Rewrite the two record texts ("name | address") of one pair the same way."""
    form = _FR_FORMS[rng.integers(len(_FR_FORMS))]
    out = []
    for text in (a, b):
        name, sep, addr = text.partition(" | ")
        name = _legal(name, form, rng)
        for pat, rep in _WORDS:
            name = pat.sub(rep, name)
        if type_words:
            for pat, rep in _TYPES:
                name = pat.sub(rep, name)
        out.append(name + sep + addr)
    return out[0], out[1]
