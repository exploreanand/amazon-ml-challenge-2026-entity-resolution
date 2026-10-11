# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** CodeCatalyst  
**Team Members:** Devashree Prashant Palav, Abhinav Anand, Aayush Pawar, Anusha Sarkar. All from K. J. Somaiya College of Engineering (KJSCE), Vidyavihar  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We resolve each Source 1 record to its Source 2/3 matches in four stages:

1. **Blocking.** Four exact GPU searches run in the *reverse* direction: every S2/S3 record retrieves its closest S1 records. They cover a multilingual embedding, character n-grams of name+address, consonant skeletons, and the address alone.
2. **Scoring.** A gradient-boosted matcher scores every candidate pair using string, number and "competition" features.
3. **Re-scoring.** Three fine-tuned cross-encoders (`bge-reranker-v2-m3`, `multilingual-e5-large` and `Qwen3-Reranker-0.6B`) re-score the uncertain pairs, and a logistic blend combines them with XGBoost. All three were trained with French-style copies of the training pairs, since France is absent from training.
4. **Rules checked on training labels:**
   - the cross-encoders may veto a pair XGBoost is sure about, but only when the core names really differ;
   - a pair is rejected when the S2/S3 name adds the record's country word;
   - an address-less record goes to the only S1 of its country with the same core name (the unique-name rule).
5. **Decision.** Each S2/S3 record keeps its best S1 if the blended probability clears the cut-off: 0.85 for US and India (validation), 0.90 for France (leaderboard probes).

The reverse search exploits the fact that each S2/S3 record belongs to at most one S1. With "sibling bridging" for records that have no usable address, it reaches **98.25% pair recall at 16.4 candidates per S1** on held-out training S1s. Its candidate lists grow with the true cluster size.

**Final submission: v11a_fr090, public leaderboard F0.5 0.984852** (validation 0.98782).

---

## 2. Methodology

### 2.1 Problem Analysis

Key findings from EDA on the training data (2.2M S1, 10.3M S2/S3):

**Cluster structure**
- 5.6% of S1 records are singletons.
- The rest have 3.7 matches on average, up to 11, split evenly between S2 and S3.
- **Every S2/S3 record matches at most one S1.** 26% of S2/S3 records match nothing.
- **All 7.6M true pairs share the country label**, so the search is partitioned by the country values found in the data. No country list is hard-coded, and France, unseen in training, is handled the same way.

**Names are not identifying**
- 45% of India S1 names are shared by another S1.
- Name-only blocking reached only 66% recall at 100 candidates per S1, so the address is essential.

**Noise seen in names**
- Letter/digit look-alikes: `5upreme`, `Enc1ave`.
- Typos and reordered words; legal suffixes moved or re-spelled (`L.L.C.`).
- `DBA`/`fka` prefixes, and acronyms (`TP` for `Training Parents`).
- Unrelated trade names at the same address.

**Noise seen in addresses**
- Letter case; zero-padded or ranged numbers (`012`, `T-006`, `183-185`).
- Street-type abbreviations (`Rd`, `St`, `Bd`, `Av`, `All.`); state/department vs region names.
- Missing components; empty addresses (3% of S2/S3).

**Scripts**
- 24% of India S2 names (13% of S3) are written in one of nine Indic scripts: Devanagari, Tamil, Telugu, Kannada, Gujarati, Bengali, Malayalam, Oriya, Gurmukhi.
- S1 is always in Latin script.

**Hard negatives**
- Many non-matches are near-copies of an S1 record: a few letters or one house number differ (`Pranita Media, Shop No 14` vs `Praniti Medai, No 14`).
- Many true matches carry similar noise, so small differences must be judged in context.

**Test-set shift**
- Test has 5.75 S2/S3 records per S1 against 4.68 in train, while strong candidates per S1 are the same. This points to more S2/S3 records whose S1 is absent.
- Leaderboard probes showed the test set rewards a stricter decision threshold than validation does.
- **France (15% of test, absent from train) carries the whole validation-to-leaderboard gap.** A probe submission with every France S1 predicted empty scored 0.846791. Solving the two submissions for the France share gives France ≈ 0.951 against ≈ 0.986 for US + India, which is the validation level.
- France names follow different patterns:
  - a "(France)" suffix, like India's "(India)";
  - French legal forms (SASU, EI, SNC…) and linking words (de, du, la…);
  - type words that decoys swap (Club, Comité, École, Compagnie, Société…).
- **Some patterns carry over exactly from the training countries, and they are the ones that transfer.**
  - An S2/S3 name that *adds* the country word ("Shakti (India) Finance Pvt Limited" for S1 "Shakti Finance") is never the same business: 0 of 84,907 India validation candidates.
  - Names that differ only in spelling, legal form ("SCI" / "S.C.I.") or an added "& Sons"/"Group" suffix are the same business in 99.95% of XGBoost-confident cases.

### 2.2 Solution Strategy

**Approach type:** blocking, then a GBDT pair classifier, then three cross-encoders re-scoring uncertain pairs, then three rules checked on training labels, and a cut-off per country.

**Core innovation:** *reverse blocking* combined with *one-to-one assignment*.
- Because an S2/S3 record belongs to at most one S1, it only needs to find its best few S1 candidates.
- Retrieving from the S2/S3 side gives high recall with very small k (1–2 per pass).
- Each S1's candidate list naturally grows with its true cluster size, and S1 singletons attract few candidates.
- The final decision assigns each S2/S3 record to at most one S1.

---

## 3. Candidate Generation (Blocking)

**Text normalisation.** Country-agnostic, applied identically to every record:
- HTML-entity decoding (`&amp;`).
- Indic scripts transliterated to Latin (IAST, `indic_transliteration`, MIT).
- Accent stripping, lowercasing.
- Look-alike digit→letter repair inside mixed tokens.
- Light phonetic folding (`sh`→`s`, aspirates, `ph`→`f`, doubled letters).
- A consonant *skeleton* that folds voiced/unvoiced pairs, so romanised Indic spellings meet English ones: `hotel power` / `hotala pavara` → `htl pvr`.
- (v3+) Street-type canonicalisation in English/French (`road`→`rd`, `boulevard`/`bd`→`bd`, `rue`→`r`, `saint`/`street`→`st`), and removal of `No`/`N°` markers.
- (v7) More French street and building types (`passage`, `résidence`, `bâtiment`, `lieu-dit`, `impasse`…), and removal of mail-routing codes that say nothing about location (`CEDEX 9`, `BP 123`, `CS 456`, `TSA 789`).

**Blocking keys.** Four reverse passes run inside each country partition. Each search is an exact top-k on the GPU (sparse TF-IDF matmul or fp16 dense matmul).

| Pass | Representation | k per S2/S3 record |
|---|---|---|
| e5 | `multilingual-e5-small` embedding of `name, address` (MIT, 118M params) | 2 |
| nameaddr | char 3-gram TF-IDF of normalised name + address | 2 |
| skall | char 2–3-gram TF-IDF of name + address consonant skeletons | 1 |
| naddr | char 3-gram TF-IDF of normalised address | 1 |

**Recall on held-out training S1s (India).** Reverse top-1 for each single pass:

| Configuration | Pair recall | Candidates / S1 |
|---|---|---|
| name-only TF-IDF, forward top-50 per source | 65.7% | 100 |
| nameaddr, reverse top-1 | 92.3% | 4.7 |
| e5, reverse top-1 | 92.5% | 4.7 |
| e5 + nameaddr, reverse top-1 each | 95.2% | 5.3 |
| **final 4-pass union** | **96.8%** | **15.5** |

**Sibling bridging (v5+).**
- Some S2/S3 records cannot be placed by blocking: no address, or only weak candidates (best name+address similarity < 0.5). Such a record often has a near-identical name twin in the other source that *was* placed.
- Each such record finds its top-4 name twins in the same country (char 3-gram TF-IDF of the normalised name, cosine ≥ 0.8), and inherits each twin's top-2 S1 candidates.
- New pairs get the same four blocking scores as every other pair, plus two bridge features (twin similarity, twin's score).
- Validation pair recall rose from 96.8% to **98.25%** at 16.4 candidates per S1.

**Candidate pairs generated (v5+):** 36.5M train, 36.1M test (France 5.4M).
- v1 generated 33.8M train and 33.4M test pairs (France 5.2M, India 15.9M, US 12.3M).
- `candidate_pairs.tsv` is exactly the set the matcher scores.

**Candidate-set size of the final submission (test):**

| Country | S1 records | S2/S3 records | Candidate pairs | Per S1: mean | median | 95th pct | 99th pct | S1s per S2/S3 record | Reduction vs all pairs in the country |
|---|---|---|---|---|---|---|---|---|---|
| US | 663,106 | 3,817,031 | 13.05M | 19.7 | 16 | 42 | 79 | 3.4 | 99.99948% |
| India | 809,986 | 4,717,565 | 17.67M | 21.8 | 15 | 53 | 118 | 3.7 | 99.99954% |
| France | 259,452 | 1,434,993 | 5.39M | 20.8 | 15 | 58 | 100 | 3.8 | 99.99855% |
| **All** | **1,732,544** | **9,969,589** | **36.11M** | **20.8** | **15** | **49** | **101** | **3.6** | **99.99946%** |

- Each S2/S3 record keeps only about 3.6 S1 candidates: the union of 1–2 per pass, plus bridging. An S1's list is the set of S2/S3 records that chose it, so it grows with the S1's true cluster size (test has 5.75 S2/S3 records per S1).
- All but 42 S1 records (0.002%) have at least one candidate.
- **A heavy tail from nearest-neighbour hubs.**
  - 194 S1 records (0.011%) have more than 1,000 candidates, holding 1.8% of all pairs. The largest has 112,027.
  - Their own records look ordinary. They are the nearest S1 for an unusually large number of S2/S3 records, the "hubness" effect of nearest-neighbour search. Without them the mean is 20.5.
  - The one-S1-per-record decision makes them harmless for accuracy, but they are wasted comparisons.
  - A per-S1 cap on the blocking score, or hub-aware retrieval (mutual nearest neighbours), would remove them. We did not change the submitted pipeline: the matcher's competition features depend on the candidate list, so the matcher would need retraining.

**How it scales:**
- Every search runs inside one country partition and keeps a fixed k per S2/S3 record, so the output grows linearly with the number of S2/S3 records.
- In the submitted code each top-k is exact: a batched GPU matrix product of each S2/S3 record's vector against the S1 vectors of its country (sparse TF-IDF or dense e5 embeddings). This is a cheap vector similarity, not the matcher, but it still touches every S1 of the country.
- At billions of records, the same passes would use approximate nearest-neighbour indexes with the same k: for example FAISS IVF/HNSW for the embeddings, and MinHash-LSH or inverted n-gram indexes for the TF-IDF passes. Nothing downstream changes.

**How true matches were protected:**
- Several complementary passes.
- Reverse retrieval, which needs only one correct S1 per record.
- Scripts transliterated before n-gram matching.
- Every pass score is recomputed for every candidate pair, so the matcher sees all similarities whichever pass found the pair.

---

## 4. Matching Model

**Features used.** 71 per pair in the final (v7) feature set. v1 had 66; the v8 experiment added 5 more, marked *(v8)* below and not used in the final submission.

*Name features*
- rapidfuzz ratio, token-set, token-sort, partial ratio; Jaro-Winkler; Levenshtein distance.
- The same on "core" names with legal/generic words removed; core-token Jaccard; skeleton similarity.
- Length of unmatched core tokens on each side; acronym match.
- *(v8, experiment)* The record's own country word is dropped from core names, taken from its country label. So "(France)" is treated as "(India)" already was, and an unseen country gets the same rule.
- *(v8, experiment)* French legal forms (SASU, EI, SELARL, SNC, SCOP, GIE) and linking words (de, du, des, la, le, les, et) count as generic.
  - Société, Compagnie, Établissements and Association stay in the core name. In this data they are type words that true variants keep and decoys swap.
- *(v8, experiment)* One-word-swap features:
  - whether the core names differ by exactly one word, and the Jaro-Winkler similarity of the two words;
  - the size of the core-word difference;
  - how common the differing words are in the partition. Type words (Club, Comité, École) are common and typos are rare, so "Club → Comité" looks like a decoy and "Martin → Martn" like a typo.

*Address features*
- Token-set and partial ratio; token Jaccard.
- Number-set Jaccard, shared numbers, numbers present on only one side, first-number equality.
- Street key: house number + street words; empty-address flag.
- (v7) Token Jaccard and token-set ratio on *rare* address tokens only. Tokens present in more than 1% of the partition's records are ignored (regions, departments, states, big cities), so "Loire-Atlantique" vs "Pays de la Loire" does not count against a pair.

*Blocking scores*
- e5 cosine and the three TF-IDF cosines for every pair, plus each pass's retrieval rank.
- (v5+) For bridged pairs: the name-twin similarity and the twin's own score.

*Competition features* (each S2/S3 record belongs to at most one S1)
- Every score's gap to the best score among the same S1's candidates and among the same S2/S3 record's candidate S1s, and its rank in each.
- Candidates per S1; S1s per candidate.

**Model type:**
- **XGBoost** (GPU, Apache-2.0), depth 9, η 0.1, 2,613 trees (the early-stopping optimum found in v3).
- (v3+) Trained on a table where 18% of training S1 records are removed, so the model sees orphaned S2/S3 records as in test.
- (v4) A **cross-encoder**: `intfloat/multilingual-e5-base` (MIT, 278M params) fine-tuned for 1 epoch on 2.7M training pairs. These are all pairs with out-of-fold XGBoost probability in (0.005, 0.995), plus 1M confident pairs.
  - It reads `name | address` of both records jointly.
  - It re-scores uncertain pairs, and a logistic blend (fitted on validation pairs it never saw) combines it with the XGBoost probability.
- (v6+) The cross-encoder is `intfloat/multilingual-e5-large` (MIT, 560M params), fine-tuned for 2 epochs on the same pairs. The checkpoint with the best validation log-loss is kept.
- (v9+) **French-style training data.** France has no labels, and the e5-large cross-encoder wrongly rejected 8% of French legal-form variants ("SCI" vs "S.C.I.", added "SASU").
  - `ber/augment_fr.py` rewrites 30% of the training pairs the same way on both records, keeping the label:
    - English legal forms become one French form per pair (SARL, SAS, SA, EURL, SASU, SCI, SNC), dotted where the original was;
    - "& Sons" → "& Fils", "Group" → "Groupe", "and" → "et", "India" → "France".
  - With `type_words` (used for the French e5-large), type words are mapped too: School → École, Committee → Comité, Pharmacy → Pharmacie, …
  - Only generic words are mapped; no external data is used.
- (v9) A second cross-encoder, `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M params), fine-tuned for 2 epochs with this augmentation (check-set logloss 0.1205, AUC 0.9901).
- (v10b) The blend uses bge plus `multilingual-e5-large` retrained the same way with type words (logloss 0.1244, AUC 0.9897). The original e5-large is dropped.
- (v11, final) A third cross-encoder: `Qwen/Qwen3-Reranker-0.6B` (Apache-2.0, 0.6B params), a small decoder language model trained to answer "yes" or "no" to whether a document fits a query.
  - The two records are the query and the document of the model card's prompt, with the instruction "judge whether they refer to the same business".
  - The pair score is logit("yes") − logit("no") at the last position, fine-tuned with binary cross-entropy (`ber/qwen_rerank.py`). This keeps the pretrained yes/no judgement instead of starting from a new classification head.
  - It was trained for 1 epoch on the same pairs, with the same French-style rewriting including type words (lr 1e-5, batch 64).
  - Check-set logloss 0.1164, AUC 0.9906: the best of our cross-encoders (bge 0.1205 / 0.9901, French-style e5-large 0.1244 / 0.9897).
  - On validation it is the strongest single model on the uncertain pairs (AUC 0.98688 against 0.98552–0.98591), and on the XGBoost-sure pairs where a veto can apply (AUC 0.99756 against 0.99409–0.99529).
  - In the final blend its weight is 0.43, against 0.13 for bge and 0.09 for the French-style e5-large. Every blend that includes it beats every blend without it on validation (0.98774–0.98782 against 0.98762–0.98768). On the leaderboard it added +0.00015.
- Share of XGBoost-sure France pairs each cross-encoder rejects (test, no labels):

| Name difference | Pairs | e5-large | bge (French-style) | e5-large (French-style) | Qwen3-Reranker (French-style) |
|---|---|---|---|---|---|
| legal form only | 207k | 8.0% | 0.17% | 0.14% | 0.27% |
| same name up to spelling | 402k | 0.50% | 0.12% | 0.11% | 0.20% |
| core names differ (mostly decoys) | 162k | 6.8% | 6.6% | 5.9% | 7.8% |

  The augmentation removed the legal-form blind spot while keeping the decoy rejections. On validation the cross-encoders are interchangeable (0.98766 / 0.98768); on the LB the French-aware blend gave +0.0012 (v9). Qwen is the strictest on the one kind a veto may act on.
- The blend is a logistic regression on [logit p_xgb, the logit of each cross-encoder, and each of those times logit p_xgb]. It is fitted on the uncertain validation pairs; the validation numbers we report come from a two-fold cross-fit over validation S1s.

**Protected cross-encoder veto and country-word rule (v7.2+).**

- *The gap.* Only uncertain pairs (0.005 < p < 0.995) reach the cross-encoder; pairs with p ≥ 0.995 are accepted on XGBoost alone. On France, unseen in training, XGBoost's confidence is unreliable.
  - v8 moved 72k France pairs across 0.995.
  - Most v8 changes were pairs where XGBoost crossed the edge and the cross-encoder disagreed, and v8 lost on the leaderboard.
- *The veto.*
  - Pairs with p ≥ 0.995 are also scored by the cross-encoder.
  - They are blended with p capped at 0.995, using the same logistic blend, so a clear "no" (cross-encoder below about 0.2) can overrule XGBoost.
- *Protection.* On validation, XGBoost-sure pairs were classified by how their names differ (`ber/variants.py`):

| Name difference (US/India validation, p ≥ 0.995) | Pairs | Truly the same business | Share of cross-encoder "no"s that were correct | France: cross-encoder "no" rate |
|---|---|---|---|---|
| same name up to spelling/case | 366k | 99.99% | 45% | 0.5% |
| legal form only ("SCI" / "S.C.I.", added "SASU") | 348k | 99.99% | 32% | 8.1% |
| suffix added ("& Sons/& Fils", "Group/Groupe", "& Associés") | 9.5k | 99.95% | — (5 vetoes) | 88% |
| core names differ ("Club" vs "Collège") | 612k | 99.9% | 82% | 8.9% |

  - The cross-encoder's "no" is reliable only when the core names differ.
  - On French legal-form and suffix variants it fails systematically: it learned English forms only.
  - So the veto is applied only to the last kind.
  - France vetoes: 47.6k without protection, 11.0k with it.
- *Country-word rule.* A pair is rejected when the S2/S3 name contains the record's country word and the S1 name does not.
  - Evidence: 0 true matches among 84,907 such India validation candidates.
  - It rejects 2,777 France candidates (1,680 of them had been accepted) and 7 in US/India.
  - The word comes from the record's country label, so the rule is country-agnostic.
- *Effect.*
  - Validation F0.5: 0.98619 → 0.98649 (India 0.98457 → 0.98494, US 0.98727 → 0.98752).
  - Test pairs removed: France 12.7k, India 1.9k, US 1.3k.
  - Public leaderboard: 0.981075 → **0.981611**.

**Unique-name rule for address-less records (v10).**
- Address-less S2/S3 records (4.4% of true pairs) caused about 70% of the remaining US/India validation loss.
  - Placing all of them perfectly would lift validation from 0.98660 to 0.99623.
  - They are 97.7% true matches, but the owner is often ambiguous because several S1s share the name.
- The matcher rejected many of them even when only one S1 in the country has that name. It knows the address is missing, but not that the name is unique.
- `ber/owner.py`: a candidate pair gets probability ≥ 0.99 when three conditions hold:
  - the S2/S3 record has no address;
  - its core name (legal words dropped, words sorted) equals its S1's;
  - no other S1 of the same country has that core name.
- On validation 96.4% of such pairs are true matches, and validation F0.5 rose 0.98660 → 0.98766.
- On test it adds 18.4k US, 12.8k India and 9.8k France pairs. A sample of new France pairs looked all genuine: "EHPAD Domicile" / "EHPAD Domicile", "Green Société SARL" / "GREEN SOCIETE SARL".
- Variants that were tested and rejected on validation:
  - a looser "clear margin" version for noisy names (newly accepted pairs only 53–66% true);
  - a name-only lookup for records blocking never retrieved (40% precision);
  - a skeleton-based key.

**Decision rule:**
1. Each S2/S3 record keeps only its highest-probability S1.
2. A pair is a match if its probability ≥ the cut-off of its country (v11: 0.85 for US and India, 0.90 for France).
3. An S1 with no pair above the cut-off is predicted as a singleton.

**Threshold selection:**
- Macro F0.5 on a hash-based 20% validation split of training S1 ids.
- It was then shifted stricter after leaderboard probes, which showed the test set favours precision:

| Threshold (v3 model) | 0.60 | 0.70 | 0.85 |
|---|---|---|---|
| Leaderboard | 0.970038 | 0.971734 | 0.972657 |

- (v11) **A cut-off per country.** Every candidate pair lies within one country, so each country's cut-off changes only that country's rows and the effects add up. This let us probe France and US/India separately on the final model:

| Final model (v11a), public LB | US/India 0.85 | US/India 0.90 |
|---|---|---|
| France 0.85 | 0.984793 | — |
| France 0.90 | **0.984852 (final)** | 0.984491 |
| France 0.95 | 0.984847 | — |

  - France gains from 0.85 to 0.90 and is flat up to 0.95. This suggests the blended probabilities are slightly over-confident on the unseen country.
  - US/India lose at 0.90 (−0.00036), as validation predicts: for the final blend, validation peaks between 0.825 and 0.85 (0.98787 at 0.825, 0.98782 at 0.85, 0.98751 at 0.90). US/India stay at 0.85.

---

## 5. Results & Error Analysis

**Scores by version** (macro F0.5; validation = held-out 20% of training S1 ids, US + India)

| Version | Change | Validation | Public LB |
|---|---|---|---|
| v1 | 4-pass reverse blocking + XGBoost | 0.9790 | 0.971677 |
| v2 | stacked 2nd-stage model (cluster features) | 0.9819 | 0.968122 |
| v3 | address canonicalisation, acronym/street features, orphan-aware training | 0.9791 | 0.971734 |
| v3.1 | v3 at threshold 0.85 | — | 0.972657 |
| v4 | + multilingual-e5-base cross-encoder on uncertain pairs | 0.9855 | 0.980415 |
| v5 | + sibling bridging (blocking recall 96.8% → 98.25%) | +0.0001 | not submitted |
| v6 | v5 + multilingual-e5-large cross-encoder, 2 epochs | 0.98618 | not submitted |
| v7 | v6 + French street types, mail-code removal, rare-token address features | 0.98619 | **0.981075** |
| v8 | v7 + France-oriented name features (country word, legal forms, one-word swaps) | 0.98627 | 0.980002 |
| v8.1 | v8 + cross-encoder veto on all XGBoost-sure pairs | 0.98656 | not submitted (vetoes 44k France pairs, mostly legal-form/suffix variants) |
| v7.1p | v7 + protected veto (only where core names differ) | 0.98649 | not submitted |
| v7.2 | v7.1p + country-word rule | 0.98649 | 0.981611 |
| v9 | v7.2 + bge-reranker-v2-m3 cross-encoder trained with French-style pairs | 0.98660 | 0.982838 |
| v10a | v9 + unique-name rule for address-less records | 0.98766 | not submitted |
| v10b | v10a with both cross-encoders French-aware (bge + retrained e5-large) | 0.98768 | 0.984644 |
| v11b | v10b with bge self-trained on France test pairs that every model agreed on | 0.98770 | not submitted (France changes mixed) |
| v11a | v10b + Qwen3-Reranker-0.6B as a third cross-encoder | 0.98782 | 0.984793 |
| v11a_t090 | v11a at cut-off 0.90 for every country | 0.98751 | 0.984491 |
| **v11a_fr090** | **v11a with cut-off 0.90 for France (final)** | **0.98782** | **0.984852** |
| v11a_fr095 | v11a with cut-off 0.95 for France | 0.98782 | 0.984847 |

**Leaderboard probes** (used only to measure, never as a final answer):

| Probe | Public LB | Reading |
|---|---|---|
| v3 model at threshold 0.60 / 0.70 / 0.85 | 0.970038 / 0.971734 / 0.972657 | test favours precision |
| v7 with every France S1 predicted empty | 0.846791 | France ≈ 0.951, US + India ≈ 0.986 |
| v8 (France rows 11% changed, US/India val flat) | 0.980002 | France fell to ≈ 0.944: most changes were France pairs that XGBoost pushed past 0.995, which skipped the cross-encoder |
| v7.2 (France rows 4.8% changed, almost all removals) | 0.981611 | France improved by roughly 0.002–0.004 |
| v9 (French-aware second cross-encoder) | 0.982838 | France ≈ 0.953 → ≈ 0.961 |
| v10b (unique-name rule + French-aware blend) | 0.984644 | about 0.001 from US/India (validation-measured), the rest France (≈ 0.969) |
| v11a (+ Qwen3-Reranker) | 0.984793 | +0.00015, in line with the validation gain on US/India (+0.00014); France about flat |
| v11a, France at 0.90 / 0.95 | 0.984852 / 0.984847 | France prefers a stricter cut-off than US/India (France ≈ 0.970) |
| v11a, every country at 0.90 | 0.984491 | US/India lose 0.00036 at 0.90, as validation predicts |

**Where the loss came from** (v1, validation, 1 − F0.5 = 0.021):

| Source | F0.5 points |
|---|---|
| true candidate rejected by the model | 0.0061 |
| true pair never retrieved by blocking | 0.0051 |
| false merge | 0.0041 |
| S1 with matches predicted empty | 0.0038 |
| singleton given a match | 0.0014 |

**Common false positives (wrong merges):**
- Near-copy decoys of an S1 record with one or two changed letters or a changed house/office number.
- Records whose true S1 is absent (orphans) attaching to a similar S1.

**Where the loss comes from now** (v10b, validation, 1 − F0.5 = 0.0123; F0.5 gained if each bucket were fixed):

| Bucket | S2/S3 record | Pairs | Gain if fixed |
|---|---|---|---|
| true pair never retrieved by blocking | no address | 17,634 | +0.0039 |
| true pair rejected by the threshold | no address | 18,923 | +0.0036 |
| true pair never retrieved by blocking | has address | 9,068 | +0.0023 |
| true pair rejected by the threshold | has address | 6,381 | +0.0013 |
| wrong merges (all kinds) | mixed | 2,642 | +0.0017 |

- Most of what remains is genuinely ambiguous: names shared by several S1s with no address to tell them apart.
- The rest is heavily garbled records whose true S1 looks less similar than several wrong ones.
- A name-only top-10 search finds the true S1 as a clear first choice for only 1.6% of the missed address-less pairs, and 0.0% of the missed pairs with an address. Wider blocking was therefore not pursued.

**Common false negatives (missed matches):**
- S2/S3 records with an empty address and a generic name.
- Indic-script names whose romanisation and address both differ strongly.
- True matches carrying the same kind of number change as decoys.

**Lessons:**
- The validation split did not track the leaderboard for model changes. The stacked model gained on validation but lost on the leaderboard.
- Leaderboard probes on the decision threshold were the most reliable signal.
- Cross-encoder capacity and epochs mattered more than extra features. On validation at threshold 0.85:
  - e5-base, 1 epoch: 0.98546
  - e5-large, 1 epoch: 0.98580
  - e5-base, 3 epochs: 0.98594
  - e5-large, 2 epochs: 0.98608
- A Qwen3-1.7B (Apache-2.0) LoRA pair classifier matched the e5-large cross-encoder on uncertain validation pairs (AUC 0.986) but added only +0.0001 to the blend, so it is not used.
- Qwen3-Reranker-0.6B, fine-tuned through its own "yes"/"no" answer, became the best cross-encoder and the largest weight in the blend (v11). It also learned faster: logloss 0.149 after 512k training pairs, where the encoder cross-encoders were at 0.163–0.205.
- The cut-off is a separate lever for an unseen country. France gained from a stricter cut-off while US/India did not. Because the countries are decided independently, each could be probed on its own.
- Self-training on France test pairs that every model agreed on (v11b) kept validation flat and changed France pairs in both directions, so it was not submitted.
- The v8 one-word-swap features lifted XGBoost alone from 0.9782 to 0.9811 on validation (threshold 0.85). After the cross-encoder blend the gain is +0.0001, because the cross-encoder already resolved most of those pairs.
- Cluster-consistency re-ranking was tested and rejected: decoys resemble a cluster's accepted members *more* than true rejects do.
- Also tested and rejected on validation:
  - per-S1 expected-F0.5 decisions: they only mimic a looser threshold, which the LB does not reward;
  - an Indian-address NER (`shiprocket-ai/open-tinybert-indian-address-ner`, Apache-2.0) for address-part features: no gain, since 35% of the hard India records have no address and pincodes are rare;
  - looser versions of the unique-name rule;
  - wider blocking for missed pairs.
- For the unseen country, better features did not help. v8 improved validation but lost on the leaderboard: new features change *where* a model is confident, and on unseen data that confidence is not calibrated.
- What worked instead was **transferring label-checked patterns**:
  - which kinds of name difference each model judges reliably;
  - the country-word pattern, never a match in training.
  These were applied as narrow rules and measured on validation first.

---

## 6. Conclusion

- Reverse blocking over four complementary representations gives near-complete recall with small, cluster-sized candidate lists.
- A GBDT on similarity, number and competition features resolves most pairs.
- A fine-tuned multilingual cross-encoder is the main lever on the hard typo-vs-decoy pairs.
- Country-agnostic normalisation (transliteration, skeletons, street-type canonicalisation) lets the same pipeline run on France without French training data.
- For an unseen country, the last points came from two sources:
  - rules checked on training labels: a cross-encoder veto limited to real name differences, and the country-word rule;
  - teaching the cross-encoders the country's forms with French-style copies of the training pairs.

  More features for that country did not help.
- The largest single late gain was structural: an address-less record belongs to the only S1 of its country with the same core name. That turns the main ambiguity into a safe rule wherever the name is unique.
- The final points came from a stronger re-scorer, Qwen3-Reranker-0.6B (+0.00015), and a stricter cut-off for the unseen country only (+0.00006).

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:
- `src/`
- `README.md`: the exact 13-step command chain that regenerates the final output files
- `requirements.txt`, with pinned versions

Final pipeline, in order:

| Step | Script | Purpose |
|---|---|---|
| 1 | `run_pipeline.py` | Blocking (4 reverse passes) and first features for every country partition |
| 2 | `exp_stack.py --oof-only` | Out-of-fold XGBoost probabilities that pick the cross-encoder's training pairs |
| 3 | `ce_run.py` | Cross-encoder training (multilingual-e5-large, 2 epochs, lr 1e-5) |
| 4 | `add_bridge.py` | Sibling-bridging candidates |
| 5 | `rebuild_features.py --features v7` | Final pair features |
| 6 | `exp_v5.py` | XGBoost (orphan-aware) + cross-encoder blend on uncertain pairs, threshold 0.85 |
| 7 | `exp_v81.py --protect --country-rule` | Protected veto and country-word rule (v7.2) |
| 8 | `ce_run.py --augment-fr 0.3` | bge-reranker-v2-m3 cross-encoder with French-style pairs |
| 9 | `exp_v9.py` | Blend with the second cross-encoder (v9); its scores are reused in step 11 |
| 10 | `ce_run.py --augment-fr 0.3 --augment-fr-types` | multilingual-e5-large cross-encoder with French-style pairs and type words |
| 11 | `exp_v10b.py --combos "ce2,ce3"` | Blend + protected veto + country-word rule + unique-name rule (v10b); its e5-large scores are reused in step 13 |
| 12 | `ce_run.py --arch qwen --augment-fr 0.3 --augment-fr-types` | Qwen3-Reranker-0.6B with French-style pairs and type words, 1 epoch |
| 13 | `exp_v10b.py --ce4 … --combos "ce2,ce3,ce4" --thr-country France=0.90` | Final blend of the three cross-encoders + all rules + per-country cut-off; writes the final files |

Experiments kept for the record:
- `exp_v3.py`, `predict_v4.py`, `predict_stack.py`;
- `exp_v10.py`: the unique-name rule on v9 alone (v10a);
- `st_run.py`: self-training of the bge cross-encoder on France test pairs (v11b);
- `score_pairs.py`: scores and caches pairs with a trained cross-encoder; the reported run used it for step 13's Qwen scores;
- `exp_ef.py`: expected-F0.5 decisions;
- `llm_run.py`: Qwen3 LoRA, not in the final blend.

Modules in `ber/`:
- `normalize` (text)
- `blocking` (GPU top-k)
- `pipeline` (per-country candidates)
- `features`
- `match` (decision rule, output format)
- `augment` (orphan simulation)
- `cross_encoder`
- `qwen_rerank` (Qwen3-Reranker fine-tuned and scored as a pair classifier)
- `augment_fr` (French-style rewriting of training pairs)
- `variants` (name-difference kinds, country-word test)
- `owner` (unique-name rule for address-less records)
- `evaluate` (blocking recall, macro F0.5)

### B. Additional Results

Uncertain validation pairs (430k pairs with XGBoost probability in (0.005, 0.995)):

| Model | ROC-AUC |
|---|---|
| XGBoost | 0.965 |
| Cross-encoder | 0.986 |
| Cross-fitted blend | 0.992 |

The final cross-encoders on the 457k uncertain validation pairs of the bridged (v7+) candidate sets:

| Model | ROC-AUC |
|---|---|
| multilingual-e5-large (original, not in the final blend) | 0.9856 |
| bge-reranker-v2-m3, French-style | 0.9859 |
| multilingual-e5-large, French-style | 0.9855 |
| **Qwen3-Reranker-0.6B, French-style** | **0.9869** |
