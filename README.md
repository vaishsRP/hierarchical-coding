# Can you trust the percentages? Automated coding of election manifestos

Live demo: https://hierarchical-coding-pegjd3jswrpypqxydqjpad.streamlit.app/

When a model sorts text into topics, people report percentages: how much of a
manifesto is about welfare. This project checks whether those percentages
survive automated coding.

Short answer: no, but a small hand checked sample fixes them.

## Results

24 English manifestos from parties the models never saw.

| Model | Agreement (alpha) | Macro F1 | Content in the wrong topic |
|---|---|---|---|
| Sentence embeddings and logistic regression | 0.475 | **0.302** | **12.6%** |
| DeBERTa, fine tuned | 0.483 | 0.239 | 16.5% |
| DeBERTa reading the handbook definitions | **0.489** | 0.283 | 13.4% |
| LLM, zero shot (gpt-oss-120b) | 0.369 | 0.239 | 25.6%* |

*Measured on a 2,000 sentence sample; the trained models score 11% to 14% on it.

- **Errors pile up on common topics.** 32 of 63 topics are significantly off.
- **A better model gave worse percentages.** DeBERTa never predicts 17 topics.
- **The codebook's definitions help; its tree structure does not.** This holds
  in Dutch too, even with English definitions.
- **Trained models flip stance** (anti EU tagged as pro EU up to 79% of the
  time). The LLM gets stance right but the percentages worst.
- **Checking 5% by hand** brings the error from 12.6% to 0.9%.
- **Dutch** (44 manifestos) shows the same patterns.
- **Survey answers:** when the British Election Study switched from human to
  LLM coding, about 1.8% of answers changed topic, one point above normal drift.

Details in `RESULTS.md`, decisions in `SPEC.md`.

## How it works

- Data from the Manifesto Project API, duplicates removed, split by party.
- Topic tree taken from the published handbook, not the code numbers.
- Every threshold, split and metric fixed in `SPEC.md` before each step ran.
- BES survey data processed on this laptop only, as its terms require.

## Limitations

- One topic per sentence; survey coding is often multi label.
- 24 test manifestos; the demo uses the small model.
- Not yet compared with agreement between human coders.

## Run it

Needs a free Manifesto Project API key in `manifesto_apikey.txt`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers scikit-learn krippendorff mapie scipy transformers sentencepiece
python fetch_slice.py && python build_codeframe.py && python make_splits.py
python baseline_cheap.py && python aggregate_experiment.py && python ppi_aggregate.py
```

GPU runs: `kaggle/` or `jobs/`. Demo: `streamlit run app/streamlit_app.py`.

## Data

No manifesto text or BES answers are in this repo. Manifesto Project, WZB
Berlin. Fieldhouse, E., et al. (2026). British Election Study Internet Panel
Waves 1 to 31. DOI 10.5255/UKDA-SN-8202-4. Not endorsed by either.

## References

- Angelopoulos, A. N., et al. (2023). Prediction powered inference. *Science, 382*, 669 to 674.
- Forman, G. (2008). Quantifying counts and costs via classification. *Data Mining and Knowledge Discovery, 17*, 164 to 206.
- Krippendorff, K. (2004). *Content analysis* (2nd ed.). Sage.
