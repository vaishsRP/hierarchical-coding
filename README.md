# Can you trust the percentages? Automated coding of election manifestos

Live demo: _link goes here once the Streamlit app is deployed_

Researchers and analysts turn text into percentages all the time. How much of a
party manifesto is about the economy, how many survey answers mention price,
what share of complaints are about delivery. Coding that text by hand is slow,
so the obvious move is to let a model do it. This project asks a narrower
question than "how accurate is the model": when a model codes the text, are
the percentages you report at the end still right?

Short answer: no, not on their own. A model that codes 51% of sentences
correctly still puts 12.6% of each manifesto's content in the wrong category,
and more training data shrinks that error but does not remove it. Checking a
random 5% of each manifesto by hand and correcting for the gap brings it under
1%, with error margins that mostly hold.

The most striking result came from the larger model. A fine tuned DeBERTa
agrees with the experts slightly more often than the small model, yet it puts
16.5% of the content in the wrong category instead of 12.6%. It gets better by
leaning on common topics, and that is exactly what skews the percentages. The
usual fixes for this, applied afterwards, made the percentages no better or
worse. Neither model is yet good enough to reduce the hand checking itself.

Data: 111 English election manifestos from 7 countries (2014 to 2024), about
141,000 sentences, each coded by a trained expert into one of 63 policy
categories from the Manifesto Project.

This is a pilot for my MSc AI thesis, which applies the same idea to coding
open ended survey answers.

## What problem this tries to solve

Most uses of manifesto data work with shares: the share of a
manifesto about welfare, compared across parties or across elections. Survey
research does the same thing with open ended answers, where the reported
number is the share of respondents who mentioned a topic, compared across
waves. In both cases the individual labels are just a means to that number.

Accuracy per sentence is the wrong target for that job. A model can get most
sentences right and still shift the shares in a consistent direction, because
its mistakes are not random. It leans towards the categories it saw most often
in training. The individual errors do not cancel out when you add them up;
they pile up on the same side.

So the project measures three separate things, and keeps them separate on
purpose:

1. **Per sentence:** how often does the model agree with the expert coder?
2. **In aggregate:** for each manifesto, how far is the predicted share of
   each category from the true share, and in which direction?
3. **The fix:** how much human checking does it take to get trustworthy shares
   back, and how many sentences does a person still need to see?

The title question of the original spec was whether using the structure of
the codeframe (domains, categories and subcategories) helps a model. That is
still part of the plan, but the aggregate question turned out to be the more
useful one, so it leads here.

## Results so far

All numbers are on the test set: 24 manifestos from parties the model never
saw in training.

**Per sentence**

| Model | Agreement with experts (alpha) | Macro F1 | Accuracy |
|---|---|---|---|
| Always guess the most common category | -0.29 | 0.00 | 14.5% |
| TF IDF and logistic regression (reference) | 0.452 | 0.270 | 48.7% |
| **Sentence embeddings and logistic regression** | **0.475** | **0.302** | **50.6%** |
| DeBERTa v3 base, fine tuned, mean of 3 seeds | **0.483** | 0.239 | **51.4%** |

Agreement is measured with Krippendorff's alpha, the same statistic used to
report agreement between human coders, so the model can later be compared
with published human numbers on the same scheme. A coder who always gives the
same answer scores below zero on this scale, not at zero.

**In aggregate**

| Training sentences | Agreement (alpha) | Content in the wrong category, counting top guesses | Same, averaging probabilities |
|---|---|---|---|
| 4,320 | 0.420 | 19.8% | 10.5% |
| 8,641 | 0.443 | 17.4% | 9.5% |
| 21,603 | 0.463 | 15.0% | 8.8% |
| 43,207 | 0.472 | 13.4% | 8.5% |
| 86,414 | 0.475 | 12.6% | 8.3% |

Four things stand out.

1. **The error is large and systematic.** 32 of the 63 categories are off by
   an amount whose 95% interval excludes zero. Welfare State Expansion is
   counted 2.4 points too often per manifesto on average; Market Regulation
   1.6 points; Technology and Infrastructure 1.2 points.
2. **It follows how common a category is.** The size of the error correlates
   with how often a category appeared in training (Spearman between 0.51 and 0.58 at every
   data size). This is the known weakness of classifying each item and then
   counting.
3. **Different parties make it worse.** Political Corruption was 0.5% of the
   training data and 3.0% of the test manifestos, and it is undercounted by
   1.5 points. Splitting by party is what exposed this; a random split would
   have hidden it.
4. **Averaging the model's probabilities instead of counting its top guesses
   removes a third of the error for free.** It is a sensible first fix, but it
   still leaves 8.3%.

**A better model, worse percentages**

| | Small model | Fine tuned DeBERTa |
|---|---|---|
| Agreement with experts (alpha) | 0.475 | 0.483 |
| Macro F1 | 0.302 | 0.239 |
| Content in the wrong category, counting top guesses | 12.6% | 16.5% |
| Same, averaging probabilities | 8.3% | 10.2% |

The fine tuned model never predicts 17 of the 63 categories. On the rarest
third of categories its F1 is 0.03, against 0.16 for the small model, while on
the commonest third the two are level. It wins on agreement by betting on
common topics, and every such bet inflates their share. Two standard fixes
were tried afterwards, both tuned on the tuning data only. Rescaling its
confidence changed nothing that matters here. Adjusting for how common each
category is lifted macro F1 to 0.265 but cut agreement to 0.433 and made the
averaged percentages far worse, because the adjusted probabilities no longer
reflect real frequencies. Details are in `RESULTS.md`.

**The fix: check a sample by hand**

| Checked by hand per manifesto | Model only | Hand checks only | Model plus hand checks | Error margins that hold |
|---|---|---|---|---|
| 5% | 12.6% | 0.8% | 0.9% | 89% |
| 10% | 12.6% | 0.5% | 0.6% | 93% |
| 20% | 12.6% | 0.4% | 0.5% | 96% |

The correction is prediction powered inference (Angelopoulos et al., 2023). A
person codes a small random sample of each manifesto, the gap between model and
person on that sample estimates the model's bias, and the full count is
corrected by it. It removes the bias whatever the model's mistakes look like.
The last column is how often the 95% margin contains the true share, for
categories above 2% of a manifesto.

With this model the corrected numbers are no better than using the hand
checked sample alone, because the model's guesses carry too little information
at this accuracy. What the correction does buy is honest error margins: margins
from the sample alone collapse for rare categories and held only 74% of the time
at 5%. The fine tuned model gives the same picture: bias removed, margins
honest, but no less error than the hand checked sample alone.

**Human review**

Asked to be right 90% of the time and to pass anything uncertain to a person
(split conformal prediction), the model was right 90.2% of the time, but it
passed 97% of sentences on. The few it coded alone were 90.5% correct. A weak
model is honest about being unsure; it is just unsure about almost everything.
The fine tuned model did no better, at 97.6%.

## How it works

**Getting the data.** There is no maintained Python client for the Manifesto
Project, so `mp_api.py` calls their REST API directly with the standard library
and caches every response. `fetch_slice.py` pulls every annotated English and
Dutch manifesto from corpus version 2026-1, about 493,000 sentences.

**Cleaning.** Nine documents turned out to be exact copies of another one: the
Australian Liberal and National coalition manifesto filed under both parties
for three elections, and the US Republican platform of 2016 reused in 2020.
Without removing them the English data looks 11% bigger than it is, and the
same text can end up in both training and test data. Headings and uncoded text
are dropped.

**The codeframe.** The coding scheme has seven domains, 56 categories, and 32
subcategories under 12 of those categories. `build_codeframe.py` reads the
parent of every code from the structure of the published handbook (the domain
headers and the "comprised of" blocks), not from the digits of the code, and
checks the result against the Manifesto Project's own codebook. Coders must use
a subcategory when one exists, so the labels sit at different depths, and a
category like 202 Democracy is never a valid label on its own.

**Rare categories.** A category is modelled if it has at least 100 examples,
which gives 63 of 76. The cutoff sits in a natural gap: the smallest kept
category has 104 examples across 27 manifestos, the largest dropped one has 82,
and more than half of those come from a single manifesto. The dropped
categories hold 0.36% of the data and are reported separately rather than
silently ignored.

**The split.** Parties, not manifestos, are assigned to training, tuning,
calibration and test data, so no party's text appears on both sides. Parties
reuse a lot of wording between elections, and a split by manifesto would let a
model memorise it. The split is balanced within each country and fixed with a
seed; its SHA256 is recorded in `SPEC.md`.

**The models.** The cheap baseline embeds each sentence with bge small and fits
a logistic regression. The strong baseline fine tunes DeBERTa v3 base on all
63 categories at once, with no information about the hierarchy, over three
seeds on a Kaggle GPU. Structured variants that use the hierarchy come next and
must beat the strong baseline by more than the variation between seeds to count.

## Decisions made before seeing results

The easiest way to fool yourself in a project like this is to choose the
threshold, the split or the metric after seeing which one makes the result look
best. So every step's protocol is written into `SPEC.md` before that step is
run, and any change afterwards gets a dated note explaining why.

This caught one real problem. Both cheap models picked a regularisation
strength on the edge of the planned grid, which can mean the grid was too
narrow. A wider grid, chosen on the tuning data only, moved agreement from 0.475
to 0.477, so the original baseline stands. Both versions are reported.

## What I found along the way

A few things in the data were not what the documentation suggested, and each
would have quietly distorted the results.

1. **Duplicated manifestos**, described above.
2. **Three codes with two parents.** The handbook puts 202.2, 605.2 and 703.2
   under their categories. The Manifesto Project's published dataset counts them
   as uncoded instead, to stay comparable with an older handbook. This project
   follows the handbook everywhere, so its shares for 605 Law and Order differ
   from the official dataset on exactly that code.
3. **One handbook, three texts.** Version 5 of the handbook exists in a 2014
   original, a 2021 revision and a 2026 revision. The 2026 revision changed the
   definitions of 15 categories, for example moving housing into Market
   Regulation. The labels were produced before 2026, so the 2021 text is used.
4. **The downloadable category spreadsheet is not a version 5 source.** It has
   no parent column for subcategories and uses older wording for several
   categories.

## Limitations

1. **One language so far.** Everything reported is English. Dutch is downloaded
   and counted but not yet modelled.
2. **Single labels.** Each manifesto sentence gets exactly one code. Survey
   answers often need several, so the aggregate finding transfers directly but
   the modelling does not.
3. **24 test manifestos.** Error margins across manifestos are real but wide,
   and Australia makes up 36% of the test sentences.
4. **The try it box uses the cheap model.** The fine tuned model is too large
   for a free hosting tier, so the demo's guesses are rougher than the best model.
5. **The model may have seen the handbook.** Large pretrained models are
   trained on web text that likely includes the Manifesto handbook. This matters
   most for the planned language model comparison and will be stated there.
6. **Human agreement is not yet on the table.** Comparing the model with
   published agreement between human coders on this scheme is planned but not
   done, so "is 0.475 good" is still open.

## Repo map

| File | What it does |
|---|---|
| `SPEC.md` | The plan, and every decision fixed before its step ran |
| `FINDINGS.md` | Data access, label counts and the problems found in steps 1 and 2 |
| `RESULTS.md` | Detailed results for every step so far |
| `mp_api.py`, `fetch_slice.py` | Manifesto Project API client and download |
| `corpus.py` | One shared view of the data: duplicates, dropped categories, parents |
| `build_codeframe.py` | Parent and child table from the handbook |
| `count_labels.py` | Label counts per language and handbook version |
| `make_splits.py` | The split by party |
| `baseline_cheap.py`, `baseline_sensitivity.py` | Cheap baseline and its tuning check |
| `train_encoder.py`, `score_predictions.py` | Strong baseline and shared scoring |
| `aggregate_experiment.py` | Predicted versus true shares, and the learning curve |
| `conformal_abstention.py` | Human review load |
| `ppi_aggregate.py` | Correcting the shares with a hand checked sample |
| `app/` | The Streamlit demo and the small files it reads |
| `kaggle/`, `jobs/` | Running the fine tuning on a Kaggle or university GPU |

## Install and run

You need a free Manifesto Project account and API key. Put the key in
`manifesto_apikey.txt` in the project folder; it is gitignored.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers scikit-learn krippendorff mapie scipy transformers sentencepiece

python fetch_slice.py          # download, about 20 minutes
python build_codeframe.py      # needs pdftotext (ships with Git for Windows)
python count_labels.py
python make_splits.py          # prints a hash that should match SPEC.md
python baseline_cheap.py       # embeds 141,000 sentences, about 40 minutes on CPU
python aggregate_experiment.py
python conformal_abstention.py
python ppi_aggregate.py
```

The strong baseline needs a GPU. `export_modelling_set.py` writes the one file
it needs, and `kaggle/kernel_train.py` or `jobs/das5_train_encoder.sh` runs it.

To run the demo locally:

```bash
python build_app_assets.py
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Data and terms of use

The Manifesto Project does not allow its data to be redistributed without
written permission, so this repository contains no manifesto text, no
metadata and no split file. The scripts rebuild all of them from the API. What
is committed is code, the category tree without its definitions, aggregate
counts, results, and the small model behind the demo.

Data: Manifesto Project, WZB Berlin Social Science Center. See
manifesto-project.wzb.eu for the recommended citation.

## References

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction powered inference. *Science, 382*(6671), 669 to 674.
- Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal
  prediction and distribution free uncertainty quantification. *arXiv:2107.07511*.
- Forman, G. (2008). Quantifying counts and costs via classification. *Data
  Mining and Knowledge Discovery, 17*(2), 164 to 206.
- Krippendorff, K. (2004). *Content analysis: An introduction to its
  methodology* (2nd ed.). Sage.
- Werner, A., Lacewell, O., Volkens, A., Matthieß, T., Zehnter, L., & van
  Rinsum, L. (2021). *Manifesto coding instructions* (5th re-revised edition).
  WZB Berlin Social Science Center.
