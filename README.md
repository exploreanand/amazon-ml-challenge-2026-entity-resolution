# Business Entity Resolution: reproduction guide

Final submission: **v10b** (public leaderboard F0.5 0.984644).

Pipeline:
1. **Blocking**, done country by country. Four reverse passes (every S2/S3 record looks up its closest S1 records), plus sibling bridging.
2. **Pair features.**
3. **XGBoost** scores every pair.
4. **Two fine-tuned cross-encoders**, both trained with French-style copies of the training pairs, re-score the uncertain pairs: `BAAI/bge-reranker-v2-m3` and `intfloat/multilingual-e5-large`. A logistic blend combines them with XGBoost.
5. **Protected cross-encoder veto**, **country-word rule**, and the **unique-name rule** for address-less records.
6. **Output files.**

`Documentation_template.md` (next to `code/` in the submission package) explains the method and the evidence behind each step.

## Setup

```bash
pip install -r requirements.txt
```

- An **NVIDIA GPU is required.** It runs the e5 embeddings for blocking, the GPU top-k search, XGBoost (`device="cuda"`), and cross-encoder training and scoring.
- The reported run used a Google Colab runtime: RTX PRO 6000 (96 GB), 48 vCPU, 176 GB RAM, Python 3.13, CUDA 12.8. Peak RAM is about 90 GB.
- Three pretrained models are downloaded from Hugging Face on first use:
  - `intfloat/multilingual-e5-small` (MIT, 118M parameters), used for blocking;
  - `intfloat/multilingual-e5-large` (MIT, 560M parameters), fine-tuned as a cross-encoder;
  - `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M parameters), fine-tuned as a cross-encoder.
- XGBoost is Apache-2.0.
- **No external data, APIs or lookups are used.** Everything is learned from the provided training files.

## Regenerating the submission

`DATA` is the folder containing `train/` and `test/`, as shipped. Run the steps in order, from this folder. Every step writes its own folder, and a step whose output already exists is skipped or resumed. Timings are for the GPU above.

```bash
DATA=../../student_resource/dataset   # the folder with train/ and test/

# 1. Blocking: 4 reverse passes per country partition, plus first features and a v1 model (not the final one).   ~2.5 h
python src/run_pipeline.py --data $DATA --work work --out output_v1

# 2. Out-of-fold XGBoost probabilities on the training pairs. They select the cross-encoder's training pairs.   ~1 h
python src/exp_stack.py --data $DATA --work work --exp work/v2 --oof-only

# 3. Cross-encoder: multilingual-e5-large fine-tuned as a pair classifier, 2 epochs,
#    keeping the checkpoint with the best validation log-loss.   ~2 h
python src/ce_run.py --data $DATA --work work --feat work --out work/ce_large_2ep \
    --base intfloat/multilingual-e5-large --lr 1e-5 --epochs 2 --eval-n 50000

# 4. Sibling bridging: extra candidates for records that blocking could not place.   ~1 h
python src/add_bridge.py --data $DATA --feat work --out work/v5

# 5. Final pair features on the bridged candidate sets (the v7 feature set).   ~35 min
python src/rebuild_features.py --data $DATA --work work/v5 --out work/v7 --features v7

# 6. XGBoost trained on a table with 18% of S1s removed (orphan simulation), then the cross-encoder
#    on uncertain pairs (0.005 < p < 0.995) and a logistic blend, threshold 0.85.   ~45 min
python src/exp_v5.py --data $DATA --feat work/v7 --ce-model work/ce_large_2ep/model/best --out output_v7

# 7. The cross-encoder also checks pairs XGBoost is sure about, and may veto one only when the core names differ;
#    a pair is rejected when the S2/S3 name adds the country word. (This was the v7.2 submission.)   ~45 min
python src/exp_v81.py --data $DATA --feat work/v7 --run output_v7 --ce-model work/ce_large_2ep/model/best \
    --ce-cache output_v7 --protect --country-rule --out output_v72

# 8. Second cross-encoder: bge-reranker-v2-m3, with 30% of the training pairs rewritten French-style
#    (legal forms, "& Sons" -> "& Fils", "India" -> "France", ...; ber/augment_fr.py).   ~2 h
python src/ce_run.py --data $DATA --work work --feat work --out work/ce_bge_fr \
    --base BAAI/bge-reranker-v2-m3 --lr 1e-5 --epochs 2 --eval-n 50000 --augment-fr 0.3

# 9. Score the same pairs with it (v9); its scores are reused by step 11.   ~1 h
python src/exp_v9.py --data $DATA --v7 output_v7 --v72s output_v72 --v72 output_v72 \
    --ce2 work/ce_bge_fr/model/best --out output_v9

# 10. Third cross-encoder: multilingual-e5-large, French-style rewriting including type words (School -> Ecole, ...).   ~2 h
python src/ce_run.py --data $DATA --work work --feat work --out work/ce_e5l_fr \
    --base intfloat/multilingual-e5-large --lr 1e-5 --epochs 2 --eval-n 50000 --augment-fr 0.3 --augment-fr-types

# 11. Final blend of the two French-aware cross-encoders with XGBoost, protected veto, country-word rule, and the
#     unique-name rule for address-less records (ber/owner.py). Writes output_v10b/ce2+ce3/.   ~1 h
python src/exp_v10b.py --data $DATA --v7 output_v7 --v72s output_v72 --v72 output_v72 --v9 output_v9 \
    --ce3 work/ce_e5l_fr/model/best --combos "ce2,ce3" --out output_v10b
mkdir -p output && cp output_v10b/ce2+ce3/*.tsv output/
```

Result:
- `output/matching_results.tsv`: the final matches, the leaderboard file.
- `output/candidate_pairs.tsv`: the exact candidate set the model scored.

Check them with the official validator:

```bash
python ../../student_resource/utils/validate_submission.py -m output/matching_results.tsv \
    -c output/candidate_pairs.tsv -t $DATA/test --check-ids
```

**Validation.** Steps 6, 7, 9 and 11 write `val_report.json`. It holds macro F0.5 on a held-out 20% of training S1 ids (a hash split), overall and per country:

| | Validation F0.5 at threshold 0.85 | Public LB |
|---|---|---|
| v7 (step 6) | 0.98619 | 0.981075 |
| v7.2 (step 7) | 0.98649 | 0.981611 |
| v9 (step 9) | 0.98660 | 0.982838 |
| **v10b (step 11, final)** | **0.98768** (India 0.98594, US 0.98884) | **0.984644** |

**Determinism.**
- Seeds are fixed: XGBoost seed 0, the pandas sampling seeds, and the hash-based split.
- GPU training of the cross-encoder and XGBoost is not bit-exact across hardware. A rerun reproduces the scores up to small differences.
- Step 2 runs on features computed by the current code. The original run used the v1 code, so the selected cross-encoder training pairs may differ very slightly.

## Layout

The final pipeline uses the steps above and these modules:

```
src/
  run_pipeline.py        step 1: blocking and first features for every country partition (+ v1 model)
  exp_stack.py           step 2: stage-1 out-of-fold probabilities (--oof-only); the rest is the v2 stacking experiment
  ce_run.py              step 3: cross-encoder training (and scoring of the uncertain validation pairs)
  add_bridge.py          step 4: sibling-bridging candidates
  rebuild_features.py    step 5: features for existing candidate sets (--features v7 | v8)
  exp_v5.py              step 6: XGBoost + cross-encoder blend, validation report, output files
  exp_v81.py             step 7: cross-encoder veto (--protect) and country-word rule (--country-rule)
  exp_v9.py              step 9: second cross-encoder in the blend (and its scores for step 11)
  exp_v10b.py            step 11: final blend of the chosen cross-encoders + all rules
  ber/normalize.py       text normalisation: Indic transliteration, accents, OCR digits, phonetic folding,
                         consonant skeletons, street-type canonicalisation (English/French), mail codes
  ber/blocking.py        TF-IDF fitting and exact GPU top-k search (sparse and dense)
  ber/pipeline.py        per-country candidate generation (4 reverse passes)
  ber/features.py        pairwise string / number / address / competition features
  ber/match.py           feature assembly, decision rule (one S1 per S2/S3 record), output format
  ber/augment.py         orphan simulation for training
  ber/cross_encoder.py   cross-encoder training and scoring
  ber/augment_fr.py      French-style rewriting of training pairs (steps 8 and 10)
  ber/variants.py        name-variant kinds and the country-word test used by steps 7, 9 and 11
  ber/owner.py           unique-name rule for address-less records (step 11)
  ber/evaluate.py        blocking recall and the official macro F0.5
```

Experiments kept for the record. They are not needed for the final output:
- `exp_v3.py`, `predict_v4.py`: earlier versions.
- `exp_v10.py`: the unique-name rule on top of v9 alone (v10a); step 11 includes it.
- `exp_ef.py`: per-S1 expected-F0.5 decisions. On validation they did not beat a plain threshold.
- `rebuild_features.py --features v8`: France name features (LB 0.980002, below v7).
- `predict_stack.py`, `ber/stack.py`: v2 stacking, which lost on the leaderboard.
- `llm_run.py`, `ber/llm_classifier.py`: a Qwen3-1.7B LoRA pair classifier (Apache-2.0). It matched the e5-large cross-encoder but added only +0.0001, so it is not used. It needs `peft`.

