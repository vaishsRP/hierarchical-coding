# Can you trust the percentages? Automated coding of election manifestos

Live demo: https://hierarchical-coding-pegjd3jswrpypqxydqjpad.streamlit.app/

When a model sorts text into topics, the number people report is a percentage:
how much of a manifesto is about welfare, how many survey answers mention
prices. This project checks whether those percentages survive automated coding.

Short answer: no. A model that tags half the sentences correctly still files
12.6% of each manifesto under the wrong topic, and a bigger model makes it
worse. Checking 5% of each manifesto by hand brings the error under 1%. The
models already code about as well as a trained human, so the problem is not the
model but trusting any single coder's percentages.

Data: 111 English and 44 Dutch election manifestos (Manifesto Project, 63
topics), and 884,000 survey answers from the British Election Study. Pilot for
my MSc AI thesis on coding open ended survey answers.

## What I found

Test set: 24 English manifestos from parties the models never saw.

| Model | Agreement (alpha) | Macro F1 | Content in the wrong topic |
|---|---|---|---|
| Sentence embeddings and logistic regression | 0.475 | **0.302** | **12.6%** |
| DeBERTa, fine tuned | 0.483 | 0.239 | 16.5% |
| DeBERTa reading the handbook definitions | **0.489** | 0.283 | 13.4% |
| LLM, zero shot (gpt-oss-120b) | 0.369 | 0.239 | 25.6%* |

*Measured on a 2,000 sentence sample; the trained models score 11% to 14% on it.

**The percentages drift, in a predictable direction.** Mistakes do not cancel
out: models over count common topics and under count rare ones. 32 of 63 topics
are significantly off. More training data shrinks the error but it levels off.

**A better model gave worse percentages.** Fine tuned DeBERTa agrees with the
experts slightly more often, but it never predicts 17 of the 63 topics. Betting
on common topics raises its agreement and inflates their share. Standard fixes
(rescaling confidence, adjusting for topic frequency) did not help.

**The codebook's words help; its tree structure does not.** Letting the model
read the handbook definition of each topic beats plain fine tuning on every
measure, mostly by rescuing rare topics (never predicted: 19 down to 4).
Predicting the policy domain first and the topic second changes nothing. Both
results hold in Dutch, even with English definitions.

**Topic right, side wrong.** Trained models often flip stance towards the more
common side: sentences against the EU are tagged as pro EU 58% to 79% of the
time. The LLM gets them right 87% of the time, yet has the worst percentages,
because it applies the topics differently from the human coders.

**The models code like a typical human.** In a published test, trained coders
agreed with the expert master coding at a median kappa of 0.46 (Mikhaylov et
al., 2012). The models reach 0.475 to 0.489, on harder text. Human coding is
noisy too.

**A small hand check fixes the percentages.** A person codes a random sample
of each manifesto, and the gap between model and person corrects the full count
(prediction powered inference).

| Checked by hand | Model only | Model plus checks | Error margins that hold |
|---|---|---|---|
| 5% | 12.6% | 0.9% | 89% |
| 10% | 12.6% | 0.6% | 93% |
| 20% | 12.6% | 0.5% | 96% |

The catch: at this accuracy the model adds little over the checked sample on
its own. It mainly buys error margins you can trust. Letting the model code
alone is not an option yet: asked to be right 90% of the time, it passes 97% of
sentences to a person.

**On short survey answers, LLM coding stays close to human coding.** The
British Election Study coded "most important issue" answers by hand until 2023,
then switched to an LLM. Tracking 116,000 respondents over 30 waves, the switch
moved about 1.8% of answers to another topic, one point above normal drift.
One or two word answers are much easier to code than manifesto sentences.

**Dutch repeats the pattern.** The bigger model agrees more (0.402 against
0.380) but gets the percentages worse, and a 5% hand check brings the error
from 18.8% to 0.7%.

Full numbers in `RESULTS.md`, every decision in `SPEC.md`.

## How it works

- **Data.** Pulled from the Manifesto Project API; nine duplicated manifestos
  removed, which had inflated the data by 11% and could leak into the test set.
- **Topics.** The parent of every topic comes from the published handbook, not
  from the code numbers. Topics with fewer than 100 examples are set aside.
- **Split.** By party, so no party's wording is in both training and test.
- **Models.** Embeddings with logistic regression; DeBERTa fine tuned on a
  Kaggle GPU, three seeds; two hierarchy variants; a zero shot LLM on Groq.
- **Decided in advance.** Every threshold, split and metric was written into
  `SPEC.md` before the step ran; later changes carry a dated note.
- **Survey data stays local.** BES answers never left the laptop, as the BES
  terms require.

## Limitations

- One topic per sentence; real survey coding is often multi label.
- 24 test manifestos; 36% of the test sentences are Australian.
- The human comparison uses two short test texts and an older handbook.
- BES hand coding was software assisted, and its LLM setup is not public.
- The demo runs the small model; DeBERTa is too large for free hosting.

## Run it

Needs a free Manifesto Project API key in `manifesto_apikey.txt`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers scikit-learn krippendorff mapie scipy transformers sentencepiece
python fetch_slice.py && python build_codeframe.py && python make_splits.py
python baseline_cheap.py && python aggregate_experiment.py && python ppi_aggregate.py
```

GPU runs: `kaggle/` or `jobs/`. Dutch: set `HC_LANG=dutch`. Demo:
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
- Menon, A. K., et al. (2021). Long tail learning via logit adjustment. *ICLR*.
- Mikhaylov, S., Laver, M., & Benoit, K. (2012). Coder reliability and misclassification in the human coding of party manifestos. *Political Analysis, 20*(1), 78 to 91.
