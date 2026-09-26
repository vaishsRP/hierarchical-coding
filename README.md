# Can you trust the percentages? Letting AI sort manifestos and survey answers by topic

Live demo: https://hierarchical-coding-pegjd3jswrpypqxydqjpad.streamlit.app/

Researchers and pollsters sort text into topics (social scientists call this
coding) and then report percentages: how much of a party's manifesto is about
welfare, how many voters name prices as the most important issue. This project
checks whether those percentages stay right when AI does the sorting.

Short answer: not on their own, but they can be fixed.

- The AI sorts about half of all sentences correctly. Many of its mistakes
  cancel out, but not all: what is left leans towards common topics, so 12.6%
  of an average manifesto's content ends up under the wrong topic.
- A bigger, more expensive model did not help. It was no better at sorting
  sentences and leaned even harder (16.5%).
- A person checking a small random sample removes the lean. That works when
  many documents or thousands of survey answers are counted together.
- When a major election survey switched from human to AI coding, its numbers
  moved only a little.

Data: 111 English and 44 Dutch election manifestos from the Manifesto Project
(63 topics), and 853,000 answers from the British Election Study. Pilot for my
MSc AI thesis on coding open ended survey answers.

## Lean and noise

Two kinds of error matter, and they behave differently.

- **Lean** is a mistake that repeats: the AI gives welfare too much weight in
  almost every manifesto. It shows up when you average many documents, which is
  exactly what comparisons between parties or years do.
- **Noise** is the one off error in a single manifesto. It is larger, but it
  goes in different directions from one manifesto to the next.

| English test manifestos | Lean | Error in one manifesto |
|---|---|---|
| AI alone | 12.6% | 22.8% |
| AI plus 5% checked by a person | 0.9% | 35.0% |
| AI plus 20% checked by a person | 0.5% | 19.2% |

So a hand check removes the lean, but for one short document a small sample
adds noise. The check pays off where percentages are built from many documents
or large surveys, not for a single short text. At today's accuracy the checked
sample on its own does about as well; the AI needs to get better before it
saves checking effort.

## What I found

**Mistakes lean towards common topics.** The AI gives common topics too much
weight and rare ones too little. Welfare, for example, is 12.2% of an average
manifesto, and the AI says 14.6%. More training data shrinks the lean, but it
levels off well above zero.

**A bigger model did not help.** A fine tuned DeBERTa model (a larger language
model trained on the same data) was no better at sorting sentences: its gain in
agreement with the experts is within the margin of error. It never picks 19 of
the 63 topics and leans harder, 16.5% against 12.6%.

**Reading the codebook helps.** Giving the model the experts' written definition
of each topic made it better on every count: agreement, rare topics and lean
(13.4%). One caveat: this version also works differently inside, so part of the
gain may come from that. Organising topics as a tree (policy area first, then
topic) made no difference.

**Topic right, side wrong.** The models often find the topic but pick the side
they saw more often in training: sentences against the EU are tagged as in
favour of the EU in 58% to 79% of cases. A general AI chatbot model (gpt-oss,
given only the topic names) gets the side right far more often for immigration
and the EU, but it misfiles the most content overall.

**The AI cannot work unsupervised yet.** If it may only tag the sentences it is
confident about, it hands 97% back to a person.

**Survey answers are easier.** The British Election Study coded answers to
"what is the most important issue?" by hand until 2023, then switched to an AI.
Its numbers shifted by about 1.8% of answers at the switch, against about 0.8%
of normal change between survey rounds. Part of that may be the news in an
election year rather than the AI, so the fair reading is: a small change at
most. One thing did jump: the share of answers left uncoded went from 0.6% to
7.4%.

**Dutch shows the same patterns.** The small model leans (18.8%), and giving the
larger model the definitions helps here too, even with English definitions.

**For context, people are noisy too.** In a published test, trained human coders
agreed with an expert master coding at 0.46 on a scale where 0 is chance and 1
is perfect (Mikhaylov et al., 2012). The models score 0.48 to 0.49 against the
Manifesto Project's own coders. The setups differ, so this is context, not a
head to head result.

## All models

| English test set, 24 manifestos | Agreement (0 chance, 1 perfect) | Rare topics (macro F1) | Lean |
|---|---|---|---|
| Small model (sentence embeddings) | 0.475 | **0.302** | **12.6%** |
| DeBERTa, fine tuned | 0.483 | 0.239 | 16.5% |
| DeBERTa reading the definitions | **0.489** | 0.283 | 13.4% |
| AI chatbot model, zero shot* | 0.369 | 0.239 | 25.6% |

*Measured on 2,000 random sentences, where the other models lean 11% to 14%.
Differences between models come with 95% intervals in `RESULTS.md`.

## What this means

For anyone who publishes percentages from AI coded text: do not report the raw
AI counts. Have a person check a random sample, correct the totals, and report
the margin of error. That is cheap when you are counting many documents or a
large survey, which is when percentages matter most.

## How it works

- Data from the Manifesto Project API, duplicates removed (they inflated the
  English data by 11%), split by party so no party's wording is in both
  training and test.
- Topics come from the published handbook, not from the code numbers.
- Every threshold, split and measure was written into `SPEC.md` before each
  step ran; later changes carry a dated note. Reviewer checks after the fact
  are marked as such.
- British Election Study answers never left the laptop, as the BES terms
  require.

## Limitations

- One topic per sentence; real survey coding often allows several.
- Only 24 English test manifestos (9 Dutch); 36% of English test sentences are
  Australian.
- The survey result cannot fully separate the AI switch from the news.
- The demo uses the small model; the larger ones are too big for free hosting.

## Run it

Needs Python 3.12, `pdftotext` (poppler; included with Git for Windows) and a
free Manifesto Project API key in `manifesto_apikey.txt`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers scikit-learn krippendorff "mapie>=1.0" scipy pandas transformers sentencepiece
python fetch_slice.py            # download (about 20 minutes)
python build_codeframe.py        # topic tree from the handbook
python count_labels.py
python make_splits.py            # prints a hash that should match SPEC.md
python baseline_cheap.py         # small model (about 40 minutes on CPU)
python aggregate_experiment.py
python conformal_abstention.py
python ppi_aggregate.py
python bootstrap_compare.py      # needs the DeBERTa runs below
```

- Larger models need a GPU: `python export_modelling_set.py`, then
  `kaggle/kernel_train.py` or `jobs/das5_train_encoder.sh`; score with
  `python evaluate_probs.py --name flat runs/...`.
- Dutch: set `HC_LANG=dutch` and repeat from `make_splits.py`.
- AI chatbot comparison: a free Groq key in `groq_apikey.txt`, then
  `python llm_zero_shot.py`.
- Survey data: download the BES combined panel and open ended files, then
  `python extract_bes.py && python bes_event_study.py` (`BES_DIR` points at
  the downloads).
- Demo: `python build_app_assets.py`, `pip install -r app/requirements.txt`,
  `streamlit run app/streamlit_app.py`.

## Data

No manifesto text or survey answers are in this repo. Manifesto Project, WZB
Berlin Social Science Center. Fieldhouse, E., et al. (2026). British Election
Study Internet Panel Waves 1 to 31. DOI 10.5255/UKDA-SN-8202-4. Neither
endorses this project.

## References

- Angelopoulos, A. N., et al. (2023). Prediction powered inference. *Science, 382*, 669 to 674.
- Forman, G. (2008). Quantifying counts and costs via classification. *Data Mining and Knowledge Discovery, 17*, 164 to 206.
- Krippendorff, K. (2004). *Content analysis* (2nd ed.). Sage.
- Mikhaylov, S., Laver, M., & Benoit, K. (2012). Coder reliability and misclassification in the human coding of party manifestos. *Political Analysis, 20*(1), 78 to 91.
