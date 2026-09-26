"""v8.1: let the cross-encoder veto pairs that XGBoost is very confident about.

    python src/exp_v81.py --data dataset --feat work/v8 --run output_v8 --ce-model work/ce_large_2ep/model/best \
        --ce-cache output_v7 --out output_v81

exp_v5 sends only pairs with XGBoost p in (LO, HI) to the cross-encoder and
accepts pairs above HI on XGBoost alone. On France, absent from training, that
confidence is unreliable: most v7 -> v8 France changes were pairs where XGBoost
crossed HI and the cross-encoder disagreed. Here the pairs with p >= HI are
scored too and blended with p capped at HI, so a clear "no" from the
cross-encoder (below about 0.2) overrules XGBoost. The XGBoost model, the
blender and the uncertain-band scores all come from --run; nothing is retrained.

With --protect the veto is limited to pairs whose core names differ
(ber.variants): on validation the cross-encoder's "no" is right 82% of the time
there, but under half the time on spelling, legal-form or suffix variants,
which are nearly always real matches. With --country-rule a pair is also
rejected when the S2/S3 name adds the record's country word (ber.variants).

  1. validation (US + India): F0.5 without the veto, with it, and with the
     protected veto; the blend cross-fitted on two halves of the validation
     S1s exactly as in exp_v5;
  2. test: the submission with the (protected) veto, and how many rows it changed.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from ber.evaluate import f05_macro, true_pairs
from ber.match import add_labels, decide, feature_names, to_output
from ber.normalize import basic
from ber.pipeline import _pmap
from ber.variants import adds_country, variant_kind
from exp_v5 import GRID, HI, LO, THRESHOLD, ce_scores, load_split, log, texts, z
from run_pipeline import is_val, load


def cached(*paths):
    """Cross-encoder scores already computed by earlier runs with the same model: DataFrame[s1, cand, ce]."""
    parts = [pd.read_parquet(p, columns=["s1", "cand", "ce"]).dropna() for p in paths if os.path.exists(p)]
    return pd.concat(parts).drop_duplicates(["s1", "cand"]) if parts else None


def halves(s1):
    return (pd.util.hash_pandas_object(s1, index=False).values % 2) == 0


def names(data, split):
    return pd.concat([load(f"{data}/{split}/{split}_source{k}.tsv") for k in (1, 2, 3)]).set_index("entity_id").business_name


def kinds(pairs, name):
    """ber.variants.variant_kind of every pair in DataFrame[s1, cand]."""
    ids = pd.unique(np.concatenate([pairs.s1.values, pairs.cand.values]))
    b = pd.Series(_pmap(basic, name.reindex(ids).fillna("").tolist()), index=ids)
    return np.array([variant_kind(x, y) for x, y in zip(b.reindex(pairs.s1).values, b.reindex(pairs.cand).values)])


def country_added(pairs, name, country):
    """ber.variants.adds_country for every pair in DataFrame[s1, cand]; country: the S1's country label per pair."""
    ids = pd.unique(np.concatenate([pairs.s1.values, pairs.cand.values]))
    b = pd.Series(_pmap(basic, name.reindex(ids).fillna("").tolist()), index=ids)
    word = {c: basic(c) for c in pd.unique(country)}
    return np.array([adds_country(x, y, word[c]) for x, y, c in
                     zip(b.reindex(pairs.s1).values, b.reindex(pairs.cand).values, country)], dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="work/v8")
    ap.add_argument("--run", default="output_v8", help="exp_v5 output dir with the model, blender and band scores")
    ap.add_argument("--ce-model", required=True)
    ap.add_argument("--ce-cache", nargs="*", default=[], help="earlier output dirs scored with the same --ce-model")
    ap.add_argument("--out", default="output_v81")
    ap.add_argument("--protect", action="store_true", help="veto only pairs whose core names differ (ber.variants)")
    ap.add_argument("--country-rule", action="store_true", help="reject pairs whose S2/S3 name adds the country word")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {"run": args.run, "cap": HI, "protect": args.protect, "country_rule": args.country_rule}

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    s1v = load(f"{args.data}/train/train_source1.tsv")[["entity_id", "country"]]
    s1v = s1v[is_val(s1v.entity_id.values)]
    full = add_labels(load_split(args.feat, "train"), truth)
    feats = feature_names(full)
    val = full[is_val(full.s1.values)].reset_index(drop=True)
    del full
    bst = xgb.Booster(model_file=f"{args.run}/xgb_trainsplit.json")
    bst.set_param({"device": "cuda"})
    val["p"] = bst.predict(xgb.DMatrix(val[feats].astype(np.float32)))
    val = val[["s1", "cand", "label", "p"]]

    vb = pd.read_parquet(f"{args.run}/val_band.parquet")
    chk = vb[["s1", "cand", "p"]].merge(val[["s1", "cand", "p"]], on=["s1", "cand"], suffixes=("_run", ""))
    log(f"val band: {len(vb):,} pairs in --run, {int(((val.p > LO) & (val.p < HI)).sum()):,} now; "
        f"max |p - p_run| {np.abs(chk.p - chk.p_run).max():.2e}")  # same model and features: expect 0
    hi = val[(val.p >= HI) & ~val.set_index(["s1", "cand"]).index.isin(vb.set_index(["s1", "cand"]).index)]
    hi = hi.reset_index(drop=True)
    txt = texts(args.data, "train")
    hi["ce"] = ce_scores(hi, txt, args.ce_model, cached(*[f"{d}/{f}" for d in args.ce_cache
                                                           for f in ("val_band.parquet", "val_high.parquet")]))
    hi.to_parquet(f"{args.out}/val_high.parquet")

    hb, hh = halves(vb.s1), halves(hi.s1)
    pb, ph = np.zeros(len(vb)), np.zeros(len(hi))
    for a in (True, False):  # the blender fitted on one half of the S1s scores the other half
        m = LogisticRegression(max_iter=1000).fit(z(vb.p[hb == a], vb.ce[hb == a]), vb.label[hb == a])
        pb[hb != a] = m.predict_proba(z(vb.p[hb != a], vb.ce[hb != a]))[:, 1]
        ph[hh != a] = m.predict_proba(z(np.minimum(hi.p[hh != a], HI), hi.ce[hh != a]))[:, 1]
    hk = kinds(hi, names(args.data, "train"))
    report["val_high_kinds"] = pd.Series(hk).value_counts().to_dict()
    vv = val[["s1", "cand", "p"]].merge(vb[["s1", "cand"]].assign(bl=pb), on=["s1", "cand"], how="left")
    vv = vv.merge(hi[["s1", "cand"]].assign(hv=ph, hp=np.where(hk == "other", ph, hi.p)), on=["s1", "cand"], how="left")
    pf_vp = vv.hp.fillna(vv.bl).fillna(vv.p).values
    maybe = pf_vp >= 0.5  # only pairs that could still be accepted
    ca = np.zeros(len(vv), dtype=bool)
    ca[maybe] = country_added(vv[maybe], names(args.data, "train"), vv.s1[maybe].map(s1v.set_index("entity_id").country).values)
    report["val_country_added"] = int(ca.sum())
    for name, pf in (("base", vv.bl.fillna(vv.p).values), ("veto", vv.hv.fillna(vv.bl).fillna(vv.p).values),
                     ("veto_protected", pf_vp), ("veto_protected_country", np.where(ca, 0.0, pf_vp))):
        report[f"val_{name}"] = {t: f05_macro(decide(vv, pf, t), truth, s1v.entity_id) for t in GRID}
        pred = decide(vv, pf, THRESHOLD)
        report[f"val_{name}_by_country"] = {c: f05_macro(pred, truth, g.entity_id) for c, g in s1v.groupby("country")}
        log(f"val {name}: {report[f'val_{name}']} | at {THRESHOLD} by country {report[f'val_{name}_by_country']}")
    vetoed = ph < THRESHOLD
    report["val_high"] = {"pairs": len(hi), "vetoed": int(vetoed.sum()), "vetoed_true": int((vetoed & (hi.label == 1)).sum()),
                          "vetoed_protected": int((vetoed & (hk == "other")).sum()),
                          "vetoed_protected_true": int((vetoed & (hk == "other") & (hi.label == 1)).sum())}
    log(f"val pairs with p >= {HI}: {report['val_high']}")
    del val, vv, txt

    blender = pickle.load(open(f"{args.run}/blender.pkl", "rb"))
    T = pd.read_parquet(f"{args.run}/test_probs.parquet")
    high = (T.p >= HI).values
    ttxt = texts(args.data, "test")
    T.loc[high, "ce"] = ce_scores(T[high], ttxt, args.ce_model,
                                  cached(*[f"{d}/test_probs.parquet" for d in [args.run] + args.ce_cache]))
    veto = blender.predict_proba(z(np.minimum(T.p[high], HI), T.ce[high]))[:, 1]
    if args.protect:
        tk = kinds(T[high], names(args.data, "test"))
        report["test_high_kinds"] = pd.Series(tk).value_counts().to_dict()
        veto = np.where(tk == "other", veto, T.p[high].values)
    T.loc[high, "p_final"] = veto
    t1 = load(f"{args.data}/test/test_source1.tsv")
    country = T.s1.map(t1.set_index("entity_id").country)
    if args.country_rule:
        maybe = (T.p_final >= 0.5).values
        ca = np.zeros(len(T), dtype=bool)
        ca[maybe] = country_added(T[maybe], names(args.data, "test"), country[maybe].values)
        report["test_country_added"] = pd.Series(country[ca].values).value_counts().to_dict()
        log(f"test pairs whose S2/S3 name adds the country word: {report['test_country_added']}")
        T.loc[ca, "p_final"] = 0.0
    T.to_parquet(f"{args.out}/test_probs.parquet")
    report["test_high"] = {c: {"pairs": int(high[m].sum()), "vetoed": int((high & (T.p_final < THRESHOLD).values)[m].sum())}
                           for c in country.unique() for m in [(country == c).values]}
    log(f"test pairs with p >= {HI}: {report['test_high']}")

    to_output(T[["s1", "cand"]], t1.entity_id, "candidate_entity_ids").to_csv(
        f"{args.out}/candidate_pairs.tsv", sep="\t", index=False)
    out = to_output(decide(T, T.p_final.values, THRESHOLD), t1.entity_id, "matched_entity_ids")
    out.to_csv(f"{args.out}/matching_results.tsv", sep="\t", index=False)
    prev = pd.read_csv(f"{args.run}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    changed = out.matched_entity_ids.values != prev.set_index("source1_entity_id").matched_entity_ids.reindex(out.source1_entity_id).values
    report["test_rows_changed"] = pd.Series(changed).groupby(t1.set_index("entity_id").country.reindex(out.source1_entity_id).values).mean().to_dict()
    log(f"test rows changed vs {args.run}: {report['test_rows_changed']}")
    json.dump(report, open(f"{args.out}/val_report.json", "w"), indent=2, default=float)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
