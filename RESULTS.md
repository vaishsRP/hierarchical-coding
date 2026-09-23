# Results so far

English, handbook 5, 63 modelled leaves, party-grouped split (test: 24
manifestos, 26,428 units). Every protocol below was fixed in SPEC.md before
the step ran. Raw numbers are in `results/`.

## Step 3: cheap baseline

| Test set | alpha, leaf | Macro F1 | Accuracy | alpha, category | alpha, domain |
|---|---|---|---|---|---|
| Majority class | -0.29 | 0.00 | 14.5% | -0.29 | -0.25 |
| TF-IDF + LR (reference) | 0.452 | 0.270 | 48.7% | 0.459 | 0.510 |
| **bge-small embeddings + LR** | **0.475** | **0.302** | **50.6%** | **0.485** | **0.555** |

- Nominal alpha is negative for a constant coder, so "no skill" sits below
  zero, not at zero.
- Embeddings beat bag-of-words by only 0.02 alpha.
- Per-leaf F1 runs from 0.69 (506 Education Expansion) to 0 for four leaves
  (302, 408, 507, 602.1). See `results/cheap_baseline_per_leaf.csv`.
- Both fits chose C on the edge of the fixed grid, so a dev-only
  sensitivity check widened each grid (`results/cheap_baseline_sensitivity.json`).
  With the wider grid, embeddings pick C = 0.003 (alpha 0.477, macro F1 0.289)
  and TF-IDF keeps C = 10, now inside its grid. Neither choice is on an edge
  and the test numbers barely move, so the pre-registered cheap baseline was
  not under-tuned and stays as the reference.

## Step 4: aggregate proportions (cheap baseline)

Per test manifesto, predicted versus true share of each leaf.
Classify-and-count (CC) is the headline; probability averaging (PA) is
reported alongside. Total bias is the share of probability mass
systematically put in the wrong leaf.

| Train units | alpha, leaf | Macro F1 | CC total bias | PA total bias | Spearman(bias, prevalence) |
|---|---|---|---|---|---|
| 4,320 (5%) | 0.420 | 0.197 | 0.198 | 0.105 | 0.51 |
| 8,641 (10%) | 0.443 | 0.230 | 0.174 | 0.095 | 0.56 |
| 21,603 (25%) | 0.463 | 0.272 | 0.150 | 0.088 | 0.58 |
| 43,207 (50%) | 0.472 | 0.290 | 0.134 | 0.085 | 0.57 |
| 86,414 (100%) | 0.475 | 0.302 | 0.126 | 0.083 | 0.57 |

(Rows below 100% are means of 3 seeds; seed spread is at most 0.016 in total bias.)

What this shows:

1. **The aggregate bias is large and systematic.** At full data, 12.6% of each
   document's coded content lands in the wrong leaf on average, and 32 of 63
   leaves have a bias whose 95% CI excludes zero. The spec's second failure
   condition (bias turns out small) does not hold.
2. **It is the classic classify-and-count pattern.** Bias correlates with
   prevalence (Spearman about 0.57 at every data size). Frequent leaves are
   over-predicted (504 +2.4 points, 403 +1.6, 411 +1.2) and rare ones
   under-predicted.
3. **Party shift adds to it.** 304 Political Corruption is 0.5% of train but
   3.0% of test documents, and is under-predicted by 1.5 points. A split by
   party exposes this; a document split would have hidden it.
4. **More data shrinks the bias but does not remove it.** From 5% to 100% of
   train, alpha rises 0.055 and CC total bias falls from 0.20 to 0.13. Both
   flatten between 50% and 100%. The strict form of the claim ("bias does
   not go away") holds; "bias does not shrink" does not.
5. **Averaging probabilities instead of counting argmaxes removes a third of
   the bias for free** (0.126 to 0.083), which is a cheap first fix before PPI.
6. **At domain level the errors mostly cancel**: only domain 5 (Welfare and
   Quality of Life, +2.7 points) has a CI excluding zero.

## Step 7: conformal abstention (cheap baseline, pipeline check)

90% target, calibrated on the calib split.

| Score | Coverage | Mean set size | Human review load | Auto-coded accuracy | Leaves under 80% coverage |
|---|---|---|---|---|---|
| LAC | 0.902 | 8.5 | 97.3% | 90.5% | 32 of 63 |
| APS | 0.914 | 10.4 | 94.3% | 88.1% | 27 of 63 |

Coverage holds on average, but with an alpha 0.47 model almost everything
goes to a human, and coverage is poor for half the leaves. The pipeline works;
the number that matters needs the strong model.

## Step 8: PPI for per-document shares (cheap baseline, pipeline check)

| Label budget per document | Estimator | Total bias | RMSE | 95% CI coverage (all / share at least 2%) | CI width |
|---|---|---|---|---|---|
| none | predictions only (CC) | 0.126 | 0.014 | no CI | |
| 5% | labels only | 0.008 | 0.023 | 0.68 / 0.74 | 0.038 |
| 5% | PPI | 0.009 | 0.024 | 0.84 / 0.89 | 0.048 |
| 10% | labels only | 0.005 | 0.018 | 0.76 / 0.84 | 0.032 |
| 10% | PPI | 0.006 | 0.019 | 0.88 / 0.93 | 0.041 |
| 20% | labels only | 0.004 | 0.012 | 0.84 / 0.91 | 0.025 |
| 20% | PPI | 0.005 | 0.013 | 0.93 / 0.96 | 0.033 |

- **PPI removes the bias**: 0.126 down to under 0.01 with 5% of units labelled.
- **With this model it does not beat labelling alone on error**: RMSE is the
  same. The predictions carry too little information at alpha 0.47 to
  tighten the interval. That gain depends on a stronger model.
- **PPI intervals are honest where label-only ones are not**: label-only
  intervals collapse for rare leaves and undercover badly (68% at 5%); PPI
  gets close to nominal at 10 to 20%.

## What is pending

- Step 5, strong flat baseline (DeBERTa-v3-base, 3 seeds) on a Kaggle GPU.
- Steps 7 and 8 rerun on the strong model; that is where review load and PPI
  efficiency are decided.
- Step 6 structured variants, step 9 GoEmotions, and the human-ceiling
  comparison against published coder reliability.
