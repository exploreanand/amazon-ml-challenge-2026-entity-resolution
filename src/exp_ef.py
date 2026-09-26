"""v9 decision rule: per-S1 expected-F0.5 decisions instead of one global threshold.

    python src/exp_ef.py --data dataset --run output_v72 --out output_v9ef [--apply-shift 0.5]

--run is an exp_v81.py output dir: val_final.parquet (validation probabilities of
every variant) and test_probs.parquet (p_final). On validation the threshold rule
is compared with ber.match.decide_expected_f for several logit shifts, on the
plain split and with 18% of validation S1s removed (an approximate orphan check:
the probabilities are not recomputed). With --apply-shift the rule is applied to
test and the output files are written.
"""

import argparse
import json
import os

import pandas as pd

from ber.evaluate import f05_macro, true_pairs
from ber.match import decide, decide_expected_f, to_output
from exp_v5 import THRESHOLD, log
from run_pipeline import is_val, load

SHIFTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--run", default="output_v72")
    ap.add_argument("--col", default="veto_protected_country", help="val_final.parquet column matching the run's test p_final")
    ap.add_argument("--out", default="output_v9ef")
    ap.add_argument("--apply-shift", type=float, help="write test output with this shift")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    s1v = load(f"{args.data}/train/train_source1.tsv")[["entity_id", "country"]]
    s1v = s1v[is_val(s1v.entity_id.values)]
    V = pd.read_parquet(f"{args.run}/val_final.parquet", columns=["s1", "cand", args.col])
    u = pd.Series(s1v.entity_id.unique())
    h = pd.util.hash_pandas_object(u, index=False, hash_key="orphan2026sim001").values
    kept = set(u[(h % 1000) >= 180])  # same 18% as ber.augment.orphanize
    Vo = V[V.s1.isin(kept)]
    splits = {"plain": (V, s1v), "orphan": (Vo, s1v[s1v.entity_id.isin(kept)])}

    def score(rule):
        out = {}
        for name, (table, ids) in splits.items():
            pred = rule(table)
            out[name] = {"all": f05_macro(pred, truth, ids.entity_id),
                         **{c: f05_macro(pred, truth, g.entity_id) for c, g in ids.groupby("country")}}
        return out

    report = {"run": args.run, "col": args.col, f"threshold_{THRESHOLD}": score(lambda t: decide(t, t[args.col].values, THRESHOLD))}
    log(f"threshold {THRESHOLD}: {report[f'threshold_{THRESHOLD}']}")
    for s in SHIFTS:
        report[f"ef_shift_{s}"] = score(lambda t: decide_expected_f(t, t[args.col].values, shift=s))
        log(f"expected-F, shift {s}: {report[f'ef_shift_{s}']}")

    if args.apply_shift is not None:
        T = pd.read_parquet(f"{args.run}/test_probs.parquet", columns=["s1", "cand", "p_final"])
        t1 = load(f"{args.data}/test/test_source1.tsv")
        pred = decide_expected_f(T, T.p_final.values, shift=args.apply_shift)
        base = decide(T, T.p_final.values, THRESHOLD)
        report["test"] = {"shift": args.apply_shift, "pairs": len(pred), "pairs_threshold_rule": len(base)}
        to_output(T[["s1", "cand"]], t1.entity_id, "candidate_entity_ids").to_csv(
            f"{args.out}/candidate_pairs.tsv", sep="\t", index=False)
        to_output(pred, t1.entity_id, "matched_entity_ids").to_csv(
            f"{args.out}/matching_results.tsv", sep="\t", index=False)
        log(f"test: {report['test']}")
    json.dump(report, open(f"{args.out}/val_report.json", "w"), indent=2, default=float)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
