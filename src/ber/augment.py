"""Orphan simulation.

The test set has ~5.75 S2/S3 records per S1 versus 4.68 in train, while the
number of strong candidates per S1 is the same: roughly 18% of test S2/S3
records have no S1 in the file ("orphans"). An orphan still resembles some
other S1 and is the main source of false merges. Removing a random share of
S1 records from the (full) candidate table reproduces this in train, both
for validation that tracks the leaderboard and for training a model that
has seen orphans.
"""

import pandas as pd

from .features import competition_features
from .match import SCORE_COLS

COMP_COLS = SCORE_COLS + ["n_tset", "a_tset"]


def orphanize(df, drop_frac, key="orphan2026sim001"):
    """Drop ``drop_frac`` of the S1 ids (and all their pairs); recompute competition features.

    Must be applied to the whole candidate table, not a subset, so that the
    competition features see every rival S1 that remains.
    """
    if drop_frac <= 0:
        return df, set(df.s1.unique())
    u = df.s1.unique()
    h = pd.util.hash_pandas_object(pd.Series(u), index=False, hash_key=key).values
    keep = set(u[(h % 1000) >= drop_frac * 1000])
    d = df[df.s1.isin(keep)].copy()
    cf = competition_features(d, COMP_COLS)
    d[cf.columns] = cf.values
    return d.reset_index(drop=True), keep
