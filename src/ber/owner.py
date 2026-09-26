"""Owners of address-less records (v10).

An S2/S3 record with no address can only be placed by its name. When exactly one
S1 of its country has the same core name (legal words dropped, words sorted),
that S1 is almost always the owner: on US/India validation 96.4% of such
candidate pairs are true matches. The matcher rejected many of them (38% of the
unique-name ones fell below the threshold) because address-less pairs are risky
in general: the same name often belongs to several S1s, and then it is a guess.
"""

import numpy as np
import pandas as pd

from .features import core_tokens
from .normalize import norm_name
from .pipeline import _pmap


def core_key(name):
    return " ".join(sorted(core_tokens(norm_name(name))))


def unique_name_pairs(pairs, rec, procs=None):
    """Mask over pairs[s1, cand]: the S2/S3 record has no address, and its core name equals its
    S1's, which no other S1 of the same country shares.

    rec: every record of the split (S1, S2, S3) indexed by entity_id, with business_name,
    business_address and country.
    """
    s1 = rec[rec.index.str.startswith("S1-")]
    ids = pd.unique(np.concatenate([s1.index.values, pairs.cand.values]))
    key = pd.Series(_pmap(core_key, rec.business_name.reindex(ids).fillna("").tolist(), procs), index=ids)
    n_same = (pd.Series(s1.country.values) + "|" + pd.Series(key.reindex(s1.index).values)).value_counts()
    k1, k2 = key.reindex(pairs.s1.values).values, key.reindex(pairs.cand.values).values
    unique = (pd.Series(s1.country.reindex(pairs.s1.values).values) + "|" + pd.Series(k1)).map(n_same).values == 1
    no_addr = rec.business_address.reindex(pairs.cand.values).fillna("").str.strip().eq("").values
    return no_addr & (k1 == k2) & (k1 != "") & unique
