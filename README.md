# Business Entity Resolution — reproduction guide

Pipeline: **data → blocking (candidate generation) → pair features → XGBoost matcher → output files**.
One command runs everything:

```bash
pip install -r requirements.txt
python src/run_pipeline.py --data <dir containing train/ and test/> --work work --out output
```

Outputs:

- `output/matching_results.tsv`: final matches (the leaderboard file)
- `output/candidate_pairs.tsv`: the exact candidate set the model scored

Intermediate files (per-country feature tables, `model.json`, `model_meta.json`) go to `--work`.
A partition whose feature file already exists is skipped, so an interrupted run resumes where it stopped.

## Hardware

- An NVIDIA GPU is required for blocking: e5 encoding, sparse top-k search, and XGBoost `device="cuda"`.
- The runs we report used a Google Colab GPU runtime (RTX PRO 6000 96 GB, 48 vCPU, 176 GB RAM).
  - Train and test blocking plus features take about 2.5 h.
  - Training and scoring take about 20 min.
- Peak RAM is about 60 GB.

## Layout

```
src/
  run_pipeline.py        end-to-end entry point
  ber/normalize.py       text normalisation, Indic transliteration, phonetic folding, consonant skeletons
  ber/blocking.py        TF-IDF fitting, exact GPU top-k (sparse and dense)
  ber/pipeline.py        per-country candidate generation (4 reverse passes)
  ber/features.py        pairwise string / number / competition features
  ber/match.py           feature assembly, decision rule, output formatting
  ber/evaluate.py        blocking recall and the official macro F0.5
```

## Models and data

- The only pretrained model is `intfloat/multilingual-e5-small` (MIT licence, 118M parameters), downloaded from Hugging Face on first use.
- The matcher is XGBoost (Apache-2.0).
- No external data, APIs or lookups are used. Every learned quantity comes from the provided training files.
