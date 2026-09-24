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

## Step 5: strong flat baseline (DeBERTa v3 base)

Fine tuned on all 63 leaves, 3 seeds on Kaggle T4 GPUs, protocol as fixed in
SPEC.md. Mean and standard deviation over seeds; scored by `evaluate_probs.py`
with the same code as everything above.

| Test set | Cheap baseline | DeBERTa, raw | DeBERTa, temperature scaled | DeBERTa, temperature and frequency adjusted |
|---|---|---|---|---|
| alpha, leaf | 0.475 | **0.483** (0.002) | 0.483 | 0.433 (0.003) |
| Macro F1 | **0.302** | 0.239 (0.002) | 0.239 | 0.265 (0.005) |
| Accuracy | 50.6% | **51.4%** | 51.4% | 45.8% |
| alpha, domain | 0.555 | **0.575** | 0.575 | 0.554 |
| Wrong category, counting top guesses | **12.6%** | 16.5% | 16.5% | 16.7% |
| Wrong category, averaging probabilities | **8.3%** | 10.2% | 10.5% | 31.0% |
| Human review load at 90% | 97.3% | 97.6% | 99.0% | 100% |

What happened:

1. **The strong model is only stronger on common categories.** It never
   predicts 17 of the 63 leaves and scores F1 0 on 20. By training frequency,
   the rarest third of leaves gets F1 0.03 against 0.16 for the cheap model,
   while the commonest third is level (0.50 and 0.49). It is also overconfident:
   mean top probability 0.65 at 51% accuracy.
2. **A better model by the usual measure produced worse percentages.** Agreement
   rose from 0.475 to 0.483, but the content counted in the wrong category rose
   from 12.6% to 16.5%. Leaning on common categories is exactly what inflates
   their share. This is the project's central claim shown directly: item level
   quality and aggregate quality are different things and can move in
   opposite directions.
3. **The standard fixes do not rescue the aggregate.** Both were applied to the
   saved probabilities and tuned on dev only, as recorded in a dated note in
   SPEC.md. Temperature scaling (T about 1.2) leaves predictions unchanged and
   makes the review load worse, because softer probabilities mean larger sets.
   Adjusting for class frequency (tau 0.75 on every seed) buys macro F1 at
   the cost of agreement, and ruins the averaged shares, because the adjusted
   probabilities no longer estimate frequencies.
4. **Conformal review at 90% is impractical at this level of agreement.**
   Every model passes 97 to 100% of sentences to a person.
5. **PPI behaves as with the cheap model.** It removes the bias, and its
   intervals hold 88, 92 and 96% of the time at 5, 10 and 20% labelled, but its
   error is no lower than the hand checked sample alone (RMSE 0.024 against
   0.023 at 5%).

The practical reading: at alpha around 0.48, no per sentence adjustment gets
the percentages right, and the reliable route is correcting them with a small
hand checked sample. Whether a model can also make that sample smaller needs
agreement well above what either model reaches here.

## Step 6: using the hierarchy

Same encoder, recipe and seeds as the flat DeBERTa. A variant counts as better
only if it beats flat by more than the seed spread (rule fixed in SPEC.md).

| Test set, mean of 3 seeds | alpha | Macro F1 | Wrong category, top guess | Wrong category, averaged | Verdict |
|---|---|---|---|---|---|
| Flat DeBERTa | 0.483 | 0.239 | 16.5% | 10.2% | reference |
| Hierarchy in the output (domain, then category) | 0.484 | 0.244 | 16.9% | 10.1% | no difference |
| **Hierarchy in the input (handbook definitions)** | **0.489** | **0.283** | **13.4%** | **9.2%** | **better** |

The strict "domain first" decode of the output variant scores the same (alpha
0.482). What helps is the experts' written definitions, not the shape of the
tree. The gain is on rare categories: F1 on the rarest third rises from 0.04 to
0.12, and categories never predicted drop from 19 to 4. Stance flips barely
change.

## Dutch

Netherlands and Flanders, 62 categories, same protocol.

| Dutch test set | Cheap model | mDeBERTa (3 seeds) |
|---|---|---|
| alpha | 0.380 | **0.402** |
| Macro F1 | **0.252** | 0.195 |
| Wrong category, top guess | **18.8%** | 20.5% |
| Wrong category, averaged | 11.2% | 11.1% |
| Review load at 90% | 99.6% | 99.9% |

Checking 5% by hand brings the error from 18.8% to 0.7% (10%: 0.5%; 20%: 0.4%),
with margins holding 92 to 98% of the time for categories above 2%. Every
English pattern repeats: the bigger model agrees more and gets rare categories
and percentages worse, and the hand checked sample fixes the percentages.

## LLM, zero shot

openai/gpt-oss-120b on Groq's free tier, temperature 0, given the 63 category
names only (no definitions, no examples). Scored against the other models on
the same sentences.

| Random 2,000 English test sentences | alpha | Macro F1 | Accuracy | Wrong category, pooled |
|---|---|---|---|---|
| LLM, zero shot | 0.369 | 0.239 | 40.2% | 25.6% |
| Cheap model | 0.499 | 0.292 | 52.8% | 11.6% |
| DeBERTa flat | 0.501 | 0.224 | 53.0% | 14.1% |
| DeBERTa with definitions | **0.505** | 0.272 | **53.4%** | **11.4%** |

| All 402 test sentences with an "against" stance | LLM right / flipped | Cheap model (best trained) right / flipped |
|---|---|---|
| Against immigration | **71% / 8%** | 14% / 27% |
| Against expanding welfare | **25% / 16%** | 3% / 56% |
| Against the EU | **87% / 6%** | 15% / 58% |
| Against the military | 26% / 6% | 38% / 21% |

- The LLM is the only model that gets stance right, because it reads meaning
  instead of learning which side is common.
- It is also the worst at the percentages, by about double. It uses the
  categories differently from the Manifesto coders, so its errors are
  systematic too, just not in the direction of common topics.
- No sign of memorisation: accuracy is 40.2% before 2024 and 39.8% on 2024
  manifestos (n = 118, a weak test; the model's training data reportedly runs
  to about mid 2024).

## Survey data: the British Election Study switch to LLM coding

"What is the most important issue facing the country?", BES internet panel,
waves 1 to 30 (2014 to 2025), 116,067 people. BES coded waves 1 to 25 by hand
(software assisted) and waves 26 to 30 with an LLM, using the same 49
categories. No answer was coded both ways, so the switch is studied as an event:
a model trained on the hand coded waves is one fixed coder across all waves, and
the change in its gap to BES at the switch estimates what the LLM changed.
Everything ran locally; no BES text left the laptop.

| | Content that moved category |
|---|---|
| At the switch (waves 20 to 25 against 26 to 30) | **1.8%** (95% CI 1.7 to 2.0) |
| At fake switch points inside the hand coded waves | median 0.8%, largest 1.1% |

- The switch moved about 1.8% of reported answers between categories, roughly
  one point more than normal drift. Largest shifts: Uncoded +0.45 points,
  Coronavirus -0.28. Living costs (-0.23) is not trustworthy: it was already
  drifting before the switch.
- Compared with the manifestos (zero shot LLM: 25.6% in the wrong category),
  an LLM set up by the study team on one or two word survey answers stays close
  to human coding.
- A first run wrongly used a 3 value label as wave 31 text; wave 31 is dropped
  (dated note in SPEC.md).

## What is pending

- Dutch definitions variant (running on Kaggle).
- Dutch C sensitivity check (the Dutch cheap fit chose C on the grid edge).
- The human ceiling: these alphas against published agreement between human
  coders on the Manifesto scheme (Mikhaylov, Laver and Benoit, 2012).
