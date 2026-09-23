# Hierarchical multi-label coding: does the codeframe earn its keep?

Personal project, September to October 2026. Pilot for the ING thesis on
public data.

## What this is for

Three things, in order of how much they matter.

1. Find out whether the structured variants actually beat the flat
   baseline, before I bet a thesis on the claim that they do. If they
   don't, I want to know that now, and the thesis becomes a different
   and more interesting question about why not.
2. Have a working repo to show a supervisor. Six people have gone quiet
   on an email describing an idea. A pilot with numbers is a different
   object.
3. Learn the evaluation, which is the part I am least sure of. The
   modelling is standard. Getting agreement metrics and conformal
   coverage right is where I will waste time if I waste it, so better to
   waste it here than at ING in December.

## The corpus, and why

**Primary: the Manifesto Project corpus.**

Party election manifestos, broken into quasi-sentences, each assigned one
code from an expert-built scheme. 3.3 million coded units, 67 countries,
five handbook versions.

The reasons this is the right one:

- The coding scheme is a genuine hierarchy with three levels. Seven
  policy domains, 56 categories, and from handbook version 5 a layer of
  subcategories beneath some categories. Written by domain experts with
  definitions and coding rules per category. That is the same object as
  ING's codeframe, not a proxy for it.
- The unit is one or two sentences of natural language. Same shape as an
  open-ended survey answer.
- **The reported output is a proportion.** Every application of this data
  computes the share of quasi-sentences per category and compares across
  parties or across elections. ING computes the share of respondents per
  category and compares across waves. This is the same measurement, which
  means the aggregate bias experiment transfers directly rather than by
  analogy.
- There is published work on coder unreliability in this scheme. So I can
  ask whether a model is worse than a human coder, against a real number,
  instead of treating human labels as ground truth.
- Multilingual, so the Dutch and English split is available.

**The one real compromise: it is single label.** One code per
quasi-sentence, mutually exclusive. The ING task is multi-label. I am not
going to hide this. It affects the loss function, the metrics, and the
constrained-decoding variant. See the GoEmotions step below.

**Second corpus, for the generalisation check only: GoEmotions.** Short
comments, 27 labels, genuinely multi-label, and the annotator-level data
was released so real disagreement is measurable. Its hierarchy is shallow
and is a grouping rather than an expert codeframe, so it cannot carry the
main result. It is there to answer "does your hierarchy finding survive
when labels are not mutually exclusive," which is the first thing anyone
will ask.

**Considered and set aside: MultiEURLEX / EURLEX57K.** Multi-label and
deeply hierarchical, and zero friction to download, which is why it is
the obvious pick. Set aside because the labels come from a single
annotating office with no second rater, so there is no disagreement to
measure and no reliability ceiling, and 700-word legal documents are not
short free text. It kills the aggregate-proportion leg of the argument,
which is the more valuable leg.

## Decisions I have to make, with my current leaning

These are the actual forks. I should decide 1 to 4 before writing code,
and 5 to 7 can wait.

**1. Which slice of the Manifesto corpus.**
3.3 million units is far more than needed and mixing handbook versions
mixes label schemes. Leaning: English-language manifestos coded under
handbook version 5 only, so the subcategory level exists, and a fixed set
of countries. If that is too small, add version 4 and drop to two
hierarchy levels. Decide by counting first.

**2. How the hierarchy is defined in code.**
The category codes are numeric and the structure is partly encoded in the
numbering, for example 4xx is the economy domain. Leaning: do not infer
the tree from digits. Build an explicit parent-child table from the
published handbook and the downloadable category spreadsheet, commit it
as a file, and treat it as data. This file is the ontology, and at ING it
is the thing I would ask the team for on day one.

**3. What counts as the target level.**
Predicting 56 categories, or predicting 7 domains, or both. Leaning:
predict the leaf level and derive the parent, because that is the harder
task and the derived parent accuracy is itself a result. If leaf-level
accuracy is too low to say anything, report at domain level as well.

**4. What the flat baseline is.**
Getting this wrong is the way to fake a positive result. A weak baseline
makes the structured variants look good for no reason. Leaning: two
baselines, one cheap and one strong. Cheap is sentence embeddings plus
multinomial logistic regression. Strong is a fine-tuned encoder over all
leaf labels at once, no hierarchy information whatsoever. The structured
variants must beat the strong one to count.

**5. Which structured variants.**
Three, and they are different kinds of claim, not three flavours of one.
- *Hierarchy in the output structure:* predict domain first, then
  category conditional on the predicted domain. Tests whether decomposing
  the decision helps.
- *Hierarchy in the input:* embed the expert category definitions from
  the handbook alongside the text and score text against definitions.
  Tests whether the written knowledge helps, not just the shape.
- *Hierarchy as a constraint at decode time:* forbid outputs where a
  child is assigned without its parent. Under single labels this is
  nearly vacuous, which is one more reason for the GoEmotions step.

**6. Zero-shot and few-shot LLM.**
Worth including because it is what ING would reach for first, and if it
wins, that is the honest finding and it changes the thesis. Leaning:
include it, keep it cheap, and be careful that the model has very likely
seen the Manifesto handbook in training. Say so.

**7. Conformal abstention.**
Split conformal on a held-out calibration set, target 90 percent
coverage, then report what fraction of items have to be sent to a human
to hit it. That number, the human review load, is the one ING actually
cares about. Leaning: use MAPIE rather than implementing it, and check
coverage empirically rather than trusting it.

## Decisions fixed after steps 1 and 2

Fixed on 2026-09-23 from label counts only, before any model was trained.
Evidence in FINDINGS.md. Changing any of these later needs a dated note
here saying why, and the results under both versions.

**Slice.** English, handbook 5, corpus 2026-1. Identical documents are kept
once (the Australian coalition manifesto filed under two party ids, the US
Republican 2016 platform reused in 2020, one Australian party's 2019 text
reused in 2022). Result: 111 manifestos, 141,349 coded units. No widening.

**Target level.** Predict the leaf: the subcategory where a category has
subcategories, otherwise the category. Split categories (202, 605, ...) are
never valid labels, following the handbook rule that coders must use the
subcategories. Category and domain are derived from the leaf through
`data/codeframe_hb5_structure.csv`.

**Sparse leaves.** A leaf is modelled if it has at least 100 coded units in
the deduplicated slice. That gives 63 English leaves. The threshold sits in
a natural gap (smallest kept leaf 103.1 with 104 units across 27
manifestos; largest dropped leaf 415 with 82 units, 55% of them in one
manifesto). The 13 dropped leaves and 000 (507 units, 0.36%) are removed
from training and evaluation and reported separately as too sparse to
evaluate. Aggregate proportions are computed over modelled units only, for
both the true and the predicted side. If Dutch is modelled, the same rule is
applied to Dutch counts separately.

**Parent convention.** The handbook table is the only parent source, for
labels and for aggregate proportions alike. 202.2, 605.2 and 703.2 therefore
sit under 202, 605 and 703, while the Manifesto Project dataset folds them
into 000. Only 605.2 is affected in practice, since 202.2 and 703.2 fall
under the sparse threshold. Category-level shares for 605 will not match the
official dataset's `per605` on exactly this point.

**Handbook revision.** Definitions are the May 2021 text, which the labels
were coded against. The August 2026 revision is not used.

**Split.** `make_splits.py`, seed 20260923. Parties, not documents, are
assigned to splits, so no party's text appears on both sides. Assignment is
balanced within each country by coded units. Result: train 61%, dev 6%,
calib 14%, test 19% of units. Every modelled leaf has at least 10 test and
45 train units. Calib is not touched before step 7. `data/splits.csv` is not
committed (terms of use); its SHA-256 is
ccd2b2dfb9bad02a59862729a1749b0576ad88053bbe844a6cca77d9f6ebfbbe and
`make_splits.py` prints it on every run.

**Model selection.** Hyperparameters are chosen on dev log-loss, not dev
macro F1, because several leaves have fewer than 10 dev units. No class
reweighting in the baselines, because reweighting distorts the predicted
proportions that the aggregate experiment measures.

**Headline metrics on test.** Krippendorff's alpha (nominal, two coders:
human and model) over all modelled test units, and macro F1 over the 63
leaves. Also reported: accuracy, category and domain accuracy derived through
the table, and per-leaf F1 with its test support.

**Note, 2026-09-23, after step 3.** Both cheap-baseline fits chose a C on
the edge of the fixed grid (embeddings 0.01, the smallest; TF-IDF 10, the
largest), with dev log-loss still improving at the edge. The pre-registered
numbers stay primary. A sensitivity check extends each grid outward (embeddings
0.001 and 0.003, TF-IDF 30 and 100), selects on dev log-loss only, and is
reported next to the primary numbers. If the extended choice wins on dev,
later comparisons against the cheap baseline use both.

**Aggregate experiment (step 4).** Fixed before it was run; full protocol in
the docstring of `aggregate_experiment.py`. Per test manifesto, true versus
predicted share of each modelled leaf. Headline estimator is
classify-and-count (what a practitioner does); probability averaging is
reported alongside. Per-leaf bias is the mean signed error over test
documents with a bootstrap CI over documents. Learning curve over 5, 10, 25,
50 and 100% of train units, 3 seeds below 100%, C re-chosen on dev each time.
The claim under test is "agreement improves with data, aggregate bias does
not go away"; it fails if total bias falls roughly in step with alpha.

**Conformal abstention (step 7).** Fixed before it was run; full protocol
in `conformal_abstention.py`. Split conformal with MAPIE, calibrated on
calib, evaluated on test, 90% target, LAC and APS scores. An item is
auto-coded only if its prediction set holds exactly one leaf. Reported:
empirical coverage, human review load, accuracy of auto-coded items, and
coverage per leaf and per country, since calib and test are different
parties and the guarantee is only marginal. First run on the cheap baseline
to build the pipeline; the number that counts comes from the strong model.

**PPI (step 8).** Fixed before it was run; full protocol in
`ppi_aggregate.py`. Per test document, compare predictions only,
labels only and PPI at label budgets of 5, 10 and 20% of the document's
units (at least 10), 200 draws each. Reported: total bias, RMSE, 95% CI
coverage (all document and leaf pairs, and pairs with a true share of at
least 2%) and CI width. Same cheap-first, strong-later rule as step 7.

## How it gets evaluated

Three separate questions. Keeping them separate is most of the value.

**Per item.** Agreement with the human code, reported as Krippendorff's
alpha rather than accuracy, so it is on the same scale as the published
human-to-human numbers. Plus macro F1 so rare categories cannot be
ignored, because rare categories are where automated coding fails and
where the interesting answers live.

**In aggregate.** For each document, the true proportion per category
against the predicted proportion per category. Report the signed error,
not the absolute error, because the sign is the whole point. Then show
what happens as training data grows: per-item agreement improves, the
aggregate bias does not go away. Then apply prediction-powered inference
on a small labelled subsample and show the bias close.

**Against the human ceiling.** Compare the model's alpha against the
published coder-reliability figures for this scheme. If the model lands
inside the range where humans disagree with each other, that is a
finding, and it reframes the whole question from "is the model good
enough" to "good enough compared to what."

## Build order

Each step ends with something reportable, so an abandoned project still
leaves a result.

1. Get access, download a slice, count what exists per category and per
   language. Report the label distribution and the long tail.
2. Build the parent-child table by hand from the handbook. Commit it.
3. Cheap baseline, embeddings plus logistic regression. First alpha
   number on the board.
4. Aggregate proportion experiment on the cheap baseline. This is the
   standalone result and it is worth writing up on its own even if I stop
   here.
5. Strong flat baseline, fine-tuned encoder.
6. The three structured variants, one at a time, each against the strong
   baseline.
7. Conformal abstention, report coverage and human review load.
8. Prediction-powered inference on the aggregate.
9. GoEmotions, variants 1 and 3 only, generalisation check.
10. README with the numbers, the failure analysis, and what it means for
    the ING project.

## How I will know it failed

Worth writing down in advance so I do not talk myself into a result.

- The structured variants do not beat the strong flat baseline by more
  than run-to-run variance. Perfectly publishable as a negative result
  and it is what I should expect, since flat classifiers over a few dozen
  labels are hard to beat.
- The aggregate bias turns out to be small. Then the PPI part has nothing
  to fix and that half of the thesis weakens.
- The leaf label distribution is so skewed that nothing can be said about
  most categories.

None of these are reasons not to run it. All three are things I would
rather learn in October than in February.

## Sources to pull before starting

- Manifesto Project database and the manifestoR access route
- The handbook version 5 coding instructions and the category spreadsheet
- The coder-reliability papers, for the human ceiling numbers
- GoEmotions, for the annotator-level file rather than the aggregated one
