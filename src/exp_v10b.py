"""v10b: several cross-encoder combinations in the blend, then v10's unique-name rule.

    python src/exp_v10b.py --data dataset --ce3 work/ce_e5l_fr/model/best --out output_v10b

Cross-encoder scores for the same pairs as v9 (nothing else is retrained):
  ce   multilingual-e5-large              (output_v7 val band, output_v72s val high, output_v72 test)
  ce2  bge-reranker-v2-m3, French-style   (output_v9 caches)
  ce3  multilingual-e5-large, French-style incl. type words (scored here, cached in --out)
For each combination in --combos the logistic blend is refitted (two-fold cross-fit
on validation, as in exp_v5), then the protected veto, the country-word rule
(exp_v81.py) and the unique-name rule (ber/owner.py) are applied. Validation F0.5
is reported per combination; test files are written to --out/<combination>/.
France diagnostics: the share of XGBoost-sure France pairs each cross-encoder
rejects, per name-variant kind.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ber.evaluate import f05_macro, true_pairs
from ber.match import decide, to_output
from ber.owner import unique_name_pairs
from exp_v5 import HI, LO, THRESHOLD, log, logit, texts
from exp_v81 import country_added, halves, kinds, names
from exp_v9 import ce2_scores
from exp_v10 import records
from run_pipeline import is_val, load


def zmat(p, ces):
    lp = logit(np.asarray(p, dtype=np.float64))
    ls = [logit(np.asarray(c)) for c in ces]
    return np.column_stack([lp, *ls, *[lp * l for l in ls]])


def cached(pairs, path, col):
    return pairs[["s1", "cand"]].merge(pd.read_parquet(path).rename(columns={"ce2": col}), on=["s1", "cand"], how="left")[col].values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--v7", default="output_v7")
    ap.add_argument("--v72s", default="output_v72s")
    ap.add_argument("--v72", default="output_v72")
    ap.add_argument("--v9", default="output_v9")
    ap.add_argument("--v10", default="output_v10", help="v10a output, for the rows-changed report (optional)")
    ap.add_argument("--ce3", required=True)
    ap.add_argument("--ce2-model", help="score ce2 with this model instead of reading the --v9 caches")
    ap.add_argument("--ce4", help="optional fourth cross-encoder (scored and cached like ce3)")
    ap.add_argument("--combos", default="ce,ce2;ce2,ce3;ce,ce2,ce3")
    ap.add_argument("--out", default="output_v10b")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    combos = [c.split(",") for c in args.combos.split(";")]
    report = {"ce3": args.ce3}

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    s1v = load(f"{args.data}/train/train_source1.tsv")[["entity_id", "country"]]
    s1v = s1v[is_val(s1v.entity_id.values)]
    vb = pd.read_parquet(f"{args.v7}/val_band.parquet", columns=["s1", "cand", "label", "p", "ce"])
    vh = pd.read_parquet(f"{args.v72s}/val_high.parquet", columns=["s1", "cand", "label", "p", "ce"])
    V = pd.read_parquet(f"{args.v72s}/val_final.parquet", columns=["s1", "cand", "label", "base"])
    txt = texts(args.data, "train")
    if args.ce2_model:
        vb["ce2"] = ce2_scores(vb, txt, args.ce2_model, f"{args.out}/val_band_ce2.parquet")
        vh["ce2"] = ce2_scores(vh, txt, args.ce2_model, f"{args.out}/val_high_ce2.parquet")
    else:
        vb["ce2"] = cached(vb, f"{args.v9}/val_band_ce2.parquet", "ce2")
        vh["ce2"] = cached(vh, f"{args.v9}/val_high_ce2.parquet", "ce2")
    vb["ce3"] = ce2_scores(vb, txt, args.ce3, f"{args.out}/val_band_ce3.parquet")
    vh["ce3"] = ce2_scores(vh, txt, args.ce3, f"{args.out}/val_high_ce3.parquet")
    if args.ce4:
        vb["ce4"] = ce2_scores(vb, txt, args.ce4, f"{args.out}/val_band_ce4.parquet")
        vh["ce4"] = ce2_scores(vh, txt, args.ce4, f"{args.out}/val_high_ce4.parquet")
    del txt
    ces = [c for c in ("ce", "ce2", "ce3", "ce4") if c in vb]
    report["val_band_auc"] = {c: roc_auc_score(vb.label, vb[c]) for c in ces}
    log(f"uncertain validation pairs, AUC: {report['val_band_auc']}")

    rec = records(args.data, "train")
    hk = kinds(vh, rec.business_name)
    rule = unique_name_pairs(V, rec)
    hb, hh = halves(vb.s1), halves(vh.s1)
    country_v = V.s1.map(s1v.set_index("entity_id").country).values
    for combo in combos:
        pb, ph = np.zeros(len(vb)), np.zeros(len(vh))
        for a in (True, False):  # the blender fitted on one half of the S1s scores the other half
            m = LogisticRegression(max_iter=1000).fit(zmat(vb.p[hb == a], [vb[c][hb == a] for c in combo]), vb.label[hb == a])
            pb[hb != a] = m.predict_proba(zmat(vb.p[hb != a], [vb[c][hb != a] for c in combo]))[:, 1]
            ph[hh != a] = m.predict_proba(zmat(np.minimum(vh.p[hh != a], HI), [vh[c][hh != a] for c in combo]))[:, 1]
        ph = np.where(hk == "other", ph, vh.p.values)
        vv = V.merge(vb[["s1", "cand"]].assign(bl=pb), on=["s1", "cand"], how="left")
        vv = vv.merge(vh[["s1", "cand"]].assign(hp=ph), on=["s1", "cand"], how="left")
        pf = vv.hp.fillna(vv.bl).fillna(vv.base).values
        maybe = pf >= 0.5
        ca = np.zeros(len(vv), dtype=bool)
        ca[maybe] = country_added(vv[maybe], rec.business_name, country_v[maybe])
        pf = np.where(ca, 0.0, pf)
        pf = np.where(rule, np.maximum(pf, 0.99), pf)
        pred = decide(vv, pf, THRESHOLD)
        name = "+".join(combo)
        report[f"val_{name}"] = {"all": f05_macro(pred, truth, s1v.entity_id),
                                 **{c: f05_macro(pred, truth, g.entity_id) for c, g in s1v.groupby("country")}}
        log(f"val F0.5 [{name}] + rules: {report[f'val_{name}']}")
    del V, vv, rec

    rec = records(args.data, "test")
    T = pd.read_parquet(f"{args.v72}/test_probs.parquet", columns=["s1", "cand", "p", "ce"])
    has = T.ce.notna().values
    ttxt = texts(args.data, "test")
    for c in ces[1:]:
        T[c] = np.nan
    if args.ce2_model:
        T.loc[has, "ce2"] = ce2_scores(T[has], ttxt, args.ce2_model, f"{args.out}/test_ce2.parquet")
    else:
        T.loc[has, "ce2"] = cached(T[has], f"{args.v9}/test_ce2.parquet", "ce2")
    T.loc[has, "ce3"] = ce2_scores(T[has], ttxt, args.ce3, f"{args.out}/test_ce3.parquet")
    if args.ce4:
        T.loc[has, "ce4"] = ce2_scores(T[has], ttxt, args.ce4, f"{args.out}/test_ce4.parquet")
    del ttxt
    band, high = ((T.p > LO) & (T.p < HI)).values, (T.p >= HI).values
    tk = kinds(T[high], rec.business_name)
    country = T.s1.map(rec.country).values
    fr = country[high] == "France"
    report["france_sure_pairs_rejected"] = {k: {"pairs": int((fr & (tk == k)).sum()), **{
        c: float((T[c][high].values[fr & (tk == k)] < 0.2).mean()) for c in ces}} for k in pd.unique(tk)}
    log(f"France, XGBoost-sure pairs rejected per cross-encoder: {report['france_sure_pairs_rejected']}")
    rule = unique_name_pairs(T, rec)
    maybe_any = (T.p > LO).values
    ca_all = np.zeros(len(T), dtype=bool)
    ca_all[maybe_any] = country_added(T[maybe_any], rec.business_name, country[maybe_any])
    test_ids = load(f"{args.data}/test/test_source1.tsv").entity_id
    prev_path = f"{args.v10}/matching_results.tsv"
    prev = (pd.read_csv(prev_path, sep="\t", dtype=str, keep_default_na=False).set_index("source1_entity_id")
            if os.path.exists(prev_path) else None)
    for combo in combos:
        blender = LogisticRegression(max_iter=1000).fit(zmat(vb.p, [vb[c] for c in combo]), vb.label)
        pf = T.p.values.astype(np.float64).copy()
        pf[band] = blender.predict_proba(zmat(T.p[band], [T[c][band] for c in combo]))[:, 1]
        veto = blender.predict_proba(zmat(np.minimum(T.p[high], HI), [T[c][high] for c in combo]))[:, 1]
        pf[high] = np.where(tk == "other", veto, T.p[high].values)
        pf = np.where(ca_all & (pf >= 0.5), 0.0, pf)
        pf = np.where(rule, np.maximum(pf, 0.99), pf)
        name = "+".join(combo)
        d = f"{args.out}/{name}"
        os.makedirs(d, exist_ok=True)
        to_output(T[["s1", "cand"]], test_ids, "candidate_entity_ids").to_csv(f"{d}/candidate_pairs.tsv", sep="\t", index=False)
        out = to_output(decide(T, pf, THRESHOLD), test_ids, "matched_entity_ids")
        out.to_csv(f"{d}/matching_results.tsv", sep="\t", index=False)
        if prev is not None:
            changed = out.matched_entity_ids.values != prev.matched_entity_ids.reindex(out.source1_entity_id).values
            report[f"test_rows_changed_vs_v10a_{name}"] = pd.Series(changed).groupby(
                rec.country.reindex(out.source1_entity_id).values).mean().to_dict()
            log(f"[{name}] test rows changed vs v10a: {report[f'test_rows_changed_vs_v10a_{name}']}")
    json.dump(report, open(f"{args.out}/val_report.json", "w"), indent=2, default=float)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
