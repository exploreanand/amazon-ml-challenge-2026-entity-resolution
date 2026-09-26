"""v10: accept address-less records whose name is unique among the S1s of their country.

    python src/exp_v10.py --data dataset --run output_v9 --col v9 --out output_v10

--run is an earlier run's output dir: val_final.parquet (validation pairs with the
run's probabilities in column --col) and test_probs.parquet (p_final). Pairs that
ber.owner.unique_name_pairs picks get probability max(p, 0.99); everything else,
including the decision rule, is unchanged. Validation F0.5 is reported before and
after, overall and per country, before the test files are written.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from ber.evaluate import f05_macro, true_pairs
from ber.match import decide, to_output
from ber.owner import unique_name_pairs
from exp_v5 import THRESHOLD, log
from run_pipeline import is_val, load


def records(data, split):
    return pd.concat([load(f"{data}/{split}/{split}_source{k}.tsv") for k in (1, 2, 3)]).set_index("entity_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--run", default="output_v9")
    ap.add_argument("--col", default="v9", help="probability column of --run/val_final.parquet")
    ap.add_argument("--out", default="output_v10")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {"run": args.run}

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    s1v = load(f"{args.data}/train/train_source1.tsv")[["entity_id", "country"]]
    s1v = s1v[is_val(s1v.entity_id.values)]
    V = pd.read_parquet(f"{args.run}/val_final.parquet", columns=["s1", "cand", "label", args.col])
    m = unique_name_pairs(V, records(args.data, "train"))
    p0 = V[args.col].values
    p1 = np.where(m, np.maximum(p0, 0.99), p0)
    report["val_rule"] = {"pairs": int(m.sum()), "true": float(V.label.values[m].mean()),
                          "newly_accepted": int((m & (p0 < THRESHOLD)).sum()),
                          "newly_accepted_true": float(V.label.values[m & (p0 < THRESHOLD)].mean())}
    log(f"validation rule pairs: {report['val_rule']}")
    for name, p in (("before", p0), ("after", p1)):
        pred = decide(V, p, THRESHOLD)
        report[f"val_{name}"] = {"all": f05_macro(pred, truth, s1v.entity_id),
                                 **{c: f05_macro(pred, truth, g.entity_id) for c, g in s1v.groupby("country")}}
        log(f"val F0.5 {name}: {report[f'val_{name}']}")
    V[["s1", "cand", "label"]].assign(v10=p1).to_parquet(f"{args.out}/val_final.parquet")
    del V

    rec = records(args.data, "test")
    T = pd.read_parquet(f"{args.run}/test_probs.parquet", columns=["s1", "cand", "p_final"])
    m = unique_name_pairs(T, rec)
    before = T.p_final.values
    T["p_final"] = np.where(m, np.maximum(before, 0.99), before)
    country = T.s1.map(rec.country).values
    report["test_rule"] = {c: {"pairs": int((m & (country == c)).sum()),
                               "newly_accepted": int((m & (country == c) & (before < THRESHOLD)).sum())}
                           for c in pd.unique(country)}
    log(f"test rule pairs by country: {report['test_rule']}")
    T.to_parquet(f"{args.out}/test_probs.parquet")
    test_ids = load(f"{args.data}/test/test_source1.tsv").entity_id
    to_output(T[["s1", "cand"]], test_ids, "candidate_entity_ids").to_csv(
        f"{args.out}/candidate_pairs.tsv", sep="\t", index=False)
    to_output(decide(T, T.p_final.values, THRESHOLD), test_ids, "matched_entity_ids").to_csv(
        f"{args.out}/matching_results.tsv", sep="\t", index=False)
    json.dump(report, open(f"{args.out}/val_report.json", "w"), indent=2, default=float)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
