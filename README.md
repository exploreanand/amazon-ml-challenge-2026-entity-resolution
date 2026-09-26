# Business Entity Resolution: reproduction guide

Final submission: **v7.2** (public leaderboard F0.5 0.981611).

Pipeline:
1. **Blocking**, done country by country. Four reverse passes (every S2/S3 record looks up its closest S1 records), plus sibling bridging.
2. **Pair features.**
3. **XGBoost** scores every pair.
4. A fine-tuned **multilingual-e5-large cross-encoder** re-scores the uncertain pairs, and a logistic blend combines the two scores.
5. **Protected cross-encoder veto** and **country-word rule**.
6. **Output files.**

`Documentation_template.md` (next to `code/` in the submission package) explains the method and the evidence behind each step.

## Setup

```bash
pip install -r requirements.txt
```

- An **NVIDIA GPU is required.** It runs the e5 embeddings for blocking, the GPU top-k search, XGBoost (`device="cuda"`), and cross-encoder training and scoring.
- The reported run used a Google Colab runtime: RTX PRO 6000 (96 GB), 48 vCPU, 176 GB RAM, Python 3.13, CUDA 12.8. Peak RAM is about 90 GB.
- Two pretrained models are downloaded from Hugging Face on first use. Both are MIT-licensed:
  - `intfloat/multilingual-e5-small` (118M parameters), used for blocking;
  - `intfloat/multilingual-e5-large` (560M parameters), fine-tuned as the cross-encoder.
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
#    keeping the checkpoint with the best validation log-loss.   ~3 h
python src/ce_run.py --data $DATA --work work --feat work --out work/ce_large_2ep \
    --base intfloat/multilingual-e5-large --lr 1e-5 --epochs 2 --eval-n 50000

# 4. Sibling bridging: extra candidates for records that blocking could not place.   ~1 h
python src/add_bridge.py --data $DATA --feat work --out work/v5

# 5. Final pair features on the bridged candidate sets (the v7 feature set).   ~35 min
python src/rebuild_features.py --data $DATA --work work/v5 --out work/v7 --features v7

# 6. XGBoost trained on a table with 18% of S1s removed (orphan simulation), then the cross-encoder
#    on uncertain pairs (0.005 < p < 0.995) and a logistic blend, threshold 0.85.   ~45 min
python src/exp_v5.py --data $DATA --feat work/v7 --ce-model work/ce_large_2ep/model/best --out output_v7

# 7. Final step:
#    - the cross-encoder also checks pairs XGBoost is sure about, and may veto one only when the core names differ;
#    - a pair is rejected when the S2/S3 name adds the country word.   ~45 min
python src/exp_v81.py --data $DATA --feat work/v7 --run output_v7 --ce-model work/ce_large_2ep/model/best \
    --ce-cache output_v7 --protect --country-rule --out output
```

Result:
- `output/matching_results.tsv`: the final matches, the leaderboard file.
- `output/candidate_pairs.tsv`: the exact candidate set the model scored.

Check them with the official validator:

```bash
python ../../student_resource/utils/validate_submission.py -m output/matching_results.tsv \
    -c output/candidate_pairs.tsv -t $DATA/test --check-ids
```

**Validation.** Steps 6 and 7 write `val_report.json`. It holds macro F0.5 on a held-out 20% of training S1 ids (a hash split), overall and per country. For the final model:

| | Validation F0.5 at threshold 0.85 |
|---|---|
| v7 | 0.98619 |
| v7.2 | 0.98649 (India 0.98494, US 0.98752) |

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
  ber/normalize.py       text normalisation: Indic transliteration, accents, OCR digits, phonetic folding,
                         consonant skeletons, street-type canonicalisation (English/French), mail codes
  ber/blocking.py        TF-IDF fitting and exact GPU top-k search (sparse and dense)
  ber/pipeline.py        per-country candidate generation (4 reverse passes)
  ber/features.py        pairwise string / number / address / competition features
  ber/match.py           feature assembly, decision rule (one S1 per S2/S3 record), output format
  ber/augment.py         orphan simulation for training
  ber/cross_encoder.py   cross-encoder training and scoring
  ber/variants.py        name-variant kinds and the country-word test used by step 7
  ber/evaluate.py        blocking recall and the official macro F0.5
```

Experiments kept for the record. They are not needed for the final output:
- `exp_v3.py`, `predict_v4.py`: earlier versions.
- `predict_stack.py`, `ber/stack.py`: v2 stacking, which lost on the leaderboard.
- `llm_run.py`, `ber/llm_classifier.py`: a Qwen3-1.7B LoRA pair classifier (Apache-2.0). It matched the e5-large cross-encoder but added only +0.0001, so it is not used. It needs `peft`.

Other feature sets:
- `rebuild_features.py --features v8` (the default) builds the v8 France name features. They scored 0.980002 on the public leaderboard, below v7, and are not used in the final submission.
