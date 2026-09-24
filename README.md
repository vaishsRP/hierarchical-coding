# Can you trust the percentages? Automated coding of election manifestos

Live demo: _link goes here once the Streamlit app is deployed_

When a model codes text into categories, the number people report is a
percentage: how much of a manifesto is about welfare, how many survey answers
mention price. This project checks whether those percentages survive automated
coding.

Short answer: no. A model that codes half the sentences correctly still puts
12.6% of each manifesto in the wrong category. A larger fine tuned model agrees
with the experts more often, yet does worse on the percentages (16.5%). An LLM
is worse still. Checking 5% of each manifesto by hand fixes it, bringing the
error under 1%. The trained model results repeat in Dutch.

Data: 111 English manifestos from 7 countries (2014 to 2024), 141,000
sentences, each coded by an expert into one of 63 Manifesto Project
categories; 44 Dutch manifestos; and 884,000 British Election Study survey
answers. Pilot for my MSc AI thesis on coding open ended survey answers.

## Results

Test set: 24 manifestos from parties the models never saw.

| Model | Agreement (alpha) | Macro F1 | Accuracy | Content in the wrong category |
|---|---|---|---|---|
| Most common category | -0.29 | 0.00 | 14.5% | |
| TF IDF and logistic regression | 0.452 | 0.270 | 48.7% | |
| Sentence embeddings and logistic regression | 0.475 | **0.302** | 50.6% | **12.6%** |
| DeBERTa v3 base, fine tuned, 3 seeds | 0.483 | 0.239 | 51.4% | 16.5% |
| DeBERTa reading the handbook definitions | **0.489** | 0.283 | **51.8%** | 13.4% |
| LLM, zero shot (gpt-oss-120b) | 0.369 | 0.239 | 40.2% | 25.6%* |

*The LLM was run on a random 2,000 sentence sample, so its figure is pooled
over that sample. On the same sample the trained models score 11% to 14%.

**The errors do not cancel out.** Models over count common topics and under
count rare ones. 32 of 63 categories are significantly off.

**A better model can give worse percentages.** DeBERTa never predicts 17 of the
63 categories. Betting on common topics raises its agreement and inflates their
share. Standard fixes (rescaling confidence, adjusting for category frequency)
did not repair it.

**The codebook's words help, its shape does not.** Giving the model the
handbook definition of each category beats plain fine tuning on every measure,
mostly by rescuing rare categories (never predicted: 19 down to 4). Predicting
the domain first, then the category, changes nothing.

**Topic right, side wrong.** Trained models often flip stance towards the more
common side: "against the EU" sentences are tagged as pro EU 58% to 79% of the
time. The LLM gets them right 87% of the time, but is the worst of all on the
percentages.

**More data helps, but not enough.**

| Training sentences | 4,320 | 21,603 | 86,414 |
|---|---|---|---|
| Agreement (alpha) | 0.420 | 0.463 | 0.475 |
| Wrong category, counting top guesses | 19.8% | 15.0% | 12.6% |
| Wrong category, averaging probabilities | 10.5% | 8.8% | 8.3% |

**Hand checking a sample fixes the percentages.** Prediction powered inference
corrects the model's count using a small random sample coded by a person.

| Checked by hand | Model only | Model plus checks | Error margins that hold |
|---|---|---|---|
| 5% | 12.6% | 0.9% | 89% |
| 10% | 12.6% | 0.6% | 93% |
| 20% | 12.6% | 0.5% | 96% |

The catch: at this accuracy the model adds little over the hand checked sample
alone. It mainly buys error margins you can trust.

**The model cannot yet work unsupervised.** Asked to be right 90% of the time
and pass anything uncertain to a person, both models pass on 97% of sentences.

**On survey answers, LLM coding stays close to human coding.** The British
Election Study coded "most important issue" answers by hand until 2023, then
switched to an LLM. Tracking the switch across 116,000 respondents and 30
waves, it moved about 1.8% of answers between categories, roughly one point
more than normal year to year drift. Short survey answers are much easier to
code than manifesto sentences.

**Dutch shows the same patterns** (44 manifestos): the bigger model agrees more
(alpha 0.402 against 0.380) but gets the percentages worse, and a 5% hand check
brings the error from 18.8% to 0.7%.

Full numbers are in `RESULTS.md`.

## How it works

- **Data.** Pulled from the Manifesto Project API (`mp_api.py`,
  `fetch_slice.py`). Nine duplicated manifestos removed, which had inflated the
  data by 11% and could leak between training and test.
- **Categories.** Parents come from the published handbook, not from the code
  digits (`build_codeframe.py`). Categories with fewer than 100 examples are set
  aside, leaving 63.
- **Split.** By party, so no party's wording appears in both training and test.
- **Models.** Sentence embeddings with logistic regression, DeBERTa fine
  tuned on a Kaggle GPU (`train_encoder.py`), two hierarchy aware variants
  (`train_structured.py`), and a zero shot LLM (`llm_zero_shot.py`).
- **Survey data.** `bes_event_study.py` runs entirely on the laptop, because
  the BES terms do not allow its answers to go to third parties.
- **Decided in advance.** Every threshold, split and metric was written into
  `SPEC.md` before the step ran. Later changes carry a dated note.

## Limitations

- One label per sentence; real survey coding is often multi label.
- 24 test manifestos; Australia is 36% of test sentences.
- The demo uses the small model; DeBERTa is too large for free hosting.
- Comparison with human coder agreement is still to do.
- The BES hand coding was software assisted, and BES's LLM setup is not public.

## Run it

You need a free Manifesto Project API key in `manifesto_apikey.txt` (gitignored).

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers scikit-learn krippendorff mapie scipy transformers sentencepiece

python fetch_slice.py
python build_codeframe.py      # needs pdftotext
python make_splits.py          # hash should match SPEC.md
python baseline_cheap.py
python aggregate_experiment.py
python conformal_abstention.py
python ppi_aggregate.py
```

Fine tuning needs a GPU: see `kaggle/` or `jobs/`. Demo:
`python build_app_assets.py`, then `streamlit run app/streamlit_app.py`.

## Data and terms

The Manifesto Project does not allow redistribution, so this repo holds no
manifesto text, metadata or handbook definitions. The scripts rebuild them from
the API. Data: Manifesto Project, WZB Berlin Social Science Center.

BES answers are personal data and are never committed or sent to any service;
only aggregate shares are published. Data: Fieldhouse, E., Green, J., Evans,
G., Mellon, J., Prosser, C., Bailey, J., de Geus, R., Schmitt, H., van der Eijk,
C., Griffiths, J., & Perrett, S. (2026). British Election Study Internet Panel
Waves 1 to 31. DOI 10.5255/UKDA-SN-8202-4. This project is not endorsed by the
BES.

## References

- Angelopoulos, A. N., Bates, S., Fannjiang, C., Jordan, M. I., & Zrnic, T.
  (2023). Prediction powered inference. *Science, 382*(6671), 669 to 674.
- Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal
  prediction and distribution free uncertainty quantification. *arXiv:2107.07511*.
- Forman, G. (2008). Quantifying counts and costs via classification. *Data
  Mining and Knowledge Discovery, 17*(2), 164 to 206.
- Krippendorff, K. (2004). *Content analysis: An introduction to its
  methodology* (2nd ed.). Sage.
- Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S.
  (2021). Long tail learning via logit adjustment. *ICLR 2021*.
