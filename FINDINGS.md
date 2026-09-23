# Step 1 and 2 findings

Corpus version 2026-1, core dataset MPDS2026a, retrieved 2026-09-23.
Reproduce with `python fetch_slice.py && python build_codeframe.py && python count_labels.py`.
Full per-code counts are in `data/label_counts.csv`. The decisions these
numbers led to are fixed in SPEC.md under "Decisions fixed after steps 1 and 2".

## Access route

No maintained Python client exists. `manifestopy` (last commit 2017) and
`pymanifesto` (last release 2019) cannot fetch coded texts. `mp_api.py` calls
the REST API directly with the standard library and caches every response
under `data/raw/`. The key goes in `manifesto_apikey.txt` (gitignored) and is
sent as a header.

## Duplicate documents

Nine downloaded documents are exact copies of another one, and all counts
below keep each text once (`corpus.dedupe`). Three affect English
handbook 5:

- the Australian Liberal/National coalition manifesto is filed under both
  63620 and 63621 for 2016, 2019 and 2022 (flagged by `is_copy_of` in the
  metadata)
- the US Republican 2020 platform is the 2016 platform reused (not flagged)
- Australian party 63901's 2022 text repeats its 2019 text (not flagged)

Without deduplication the English slice looks 11% bigger than it is
(156,641 coded units instead of 141,284), and identical documents could land
on both sides of a train/test split.

## Counts

| | English | Dutch |
|---|---|---|
| Annotated manifestos, handbook 5 | 111 | 44 |
| Countries | 7 | 2 (Netherlands, Belgium) |
| All units | 162,424 | 118,383 |
| Not coded (`NA`) | 6,106 | 4,046 |
| Headings (`H`) | 14,969 | 7,174 |
| Coded 000 | 65 | 99 |
| **Coded to a category** | **141,284** | **107,064** |
| Off-scheme codes | 0 | 0 |

Handbook 5 has 76 leaves: 44 categories without subcategories plus 32
subcategories (12 categories are split, see `data/codeframe_hb5_structure.csv`).

| Leaf distribution | English | Dutch |
|---|---|---|
| Median leaf | 600 | 510 |
| Largest leaf | 504 Welfare State Expansion, 18,871 (13%) | 503 Equality, 13,466 (13%) |
| Top 10 leaves' share | 60% | 60% |
| Leaves under 100 | 13 | 14 |
| Leaves under 300 | 26 | 33 |
| Leaves under 1,000 | 48 | 49 |
| Leaves with zero | 2 (305.4, 305.5) | 5 (305.4, 305.5, 305.6, 607.3, 608.3) |
| Categories under 100, subcategories collapsed to 56 | 2 (102, 415) | 5 |

English leaves under 100: 415 (82), 606.2 (65), 102 (64), 608.2 (53),
703.2 (40), 608.3 (36), 416.1 (31), 202.2 (27), 103.2 (26), 608.1 (14),
305.6 (4), 305.5 (0), 305.4 (0).

Dutch leaves under 100: 606.2 (96), 103.2 (94), 706 (68), 507 (42),
305.2 (23), 202.2 (23), 702 (8), 705 (6), 102 (4), and zero for 305.4,
305.5, 305.6, 607.3, 608.3.

**Subcategories are used, and parent codes never are.** In both languages
none of the 12 split categories is ever assigned as a bare code, which
matches the handbook rule that coders must use the subcategories. The
subcategories hold 25% of coded units. The splits are lopsided, though:
416.2 takes 99% of 416, 703.1 takes 99% of 703, and 202.1 takes 84% of 202.
305.4 to 305.6 are for transitional regimes only, per the handbook, so their
near-zero counts are by design.

**The English slice is uneven by country.** Australia 29% of units,
UK 18%, New Zealand 18%, Ireland 15%, Canada 8%, US 7%, South Africa 5%
(one election). Elections run from 2014 to 2024.

## Recommendation

**The English handbook 5 slice is big enough. Do not widen it.** There are
141,284 coded units across 111 documents, with a median of 600 per leaf.
The spec's third failure condition (a skew so bad that nothing can be said
about most categories) does not hold: 50 of the 76 leaves have 300 or more
examples. What does hold is a tail of 10 leaves (excluding 305.4 to 305.6)
too small to evaluate individually. Widening does not fix that tail:

| English, category level | Documents | Units | Categories under 100 |
|---|---|---|---|
| Handbook 5 only | 111 | 141,284 | 2 |
| Handbooks 4 and 5 | 171 | 184,306 | 1 |
| All handbooks | 239 | 224,434 | 0 |

Adding handbook 4 buys one category and costs the subcategory level for 60
documents, and it mixes two coding schemes. That is the fallback the spec
names, and the numbers say it is not needed.

What was decided instead (details in SPEC.md): model the 63 leaves with at
least 100 units, set aside the rest, and split by party within country.

## Parent-child table

Two files, 96 rows each (7 domains, 57 categories including 000,
32 subcategories):

- `data/codeframe_hb5_structure.csv`, committed: code and parent_code. Every
  script reads parents from this file.
- `data/codeframe_hb5.csv`, local only: code, parent_code and definition.
  The definitions are handbook text, so it is not committed (see Terms of
  use). `python build_codeframe.py` regenerates it from the handbook PDF and
  refuses to write if the parsed tree differs from the committed structure.

Source: the "Category Scheme" section of *Manifesto Coding Instructions,
5th re-revised edition* (May 2021). Parents come from that section's
`DOMAIN n` headers and its `[xxx ..., comprised of:]` blocks, not from the
digits of the codes. Definitions are the handbook text verbatim, including
its coding notes. The 7 domains and 11 split categories have empty
definitions, because the handbook prints only a heading for them. The one
exception is a note under Domain 7. I did not write any definitions.

Cross-checks in `build_codeframe.py`: every code and domain matches the API
codebook. Every subcategory lands under the category whose code prefix it
shares, which is a check on the parse, not a source for it.

**Divergence from the official dataset.** Following the handbook, 202.2,
605.2 and 703.2 sit under 202, 605 and 703. The Manifesto Project dataset
folds them into 000 (`peruncod`) for comparability with handbook 4. Shares
computed here for 605 (and for 202 and 703 if they were modelled) will
therefore differ from the dataset's `per605` on exactly these codes. This is
deliberate and applies to labels and aggregate proportions alike.

## Things in SPEC.md that were wrong or open

1. **No single leaf level.** Resolved in SPEC.md: predict the leaf, 63
   modelled leaves.
2. **Three subcategories with two parents.** Resolved: handbook convention
   everywhere, see above.
3. **The "downloadable category spreadsheet" is not a version 5 source.** It
   is `codebook_categories_MPDS*.csv`, which is identical to the API
   codebook. It has no parent column for subcategories, and for several main
   categories (102, 108, 110, 405, 415) it carries older wording than the
   version 5 handbook. The handbook is used instead.
4. **"Handbook version 5" is three texts:** March 2014 (not among the
   handbooks the Manifesto Project lists), May 2021, and August 2026. The
   2026 revision changed 15 definitions, some of which change what a code
   covers: housing now appears under 403 and 411, digitalisation under 303,
   free trade agreements under 407, tourism under 410. The May 2021 text is
   used (`--revision 2026` builds the other). Documents coded before May 2021
   may have been coded against the 2014 text, which I could not retrieve;
   the metadata does not record when a document was coded.
5. **"3.3 million coded units, 67 countries" is unverified.** I only pulled
   English and Dutch. Cite it or drop it.
6. **The Dutch slice is Netherlands plus Belgium (Flemish parties).** Decide
   whether that is the comparison you want if Dutch is modelled.
7. **Headings.** 9% of English handbook 5 units are headings (`H`) with no
   code. Per-document proportions are computed over coded units only.

## Terms of use

The Manifesto Project's terms say: "Redistribution of the provided data is
forbidden except when redistribution is authorized in writing by the
Manifesto Project." Published work using the data must cite it and be sent
to manifesto-project-bibliography@wzb.eu. `data/codeframe_hb5.csv`,
`data/manifesto_metadata.csv` and `data/splits.csv` reproduce Manifesto
Project material, so they are gitignored and regenerated by the scripts.
What is committed instead: the code tree without definitions
(`data/codeframe_hb5_structure.csv`), aggregate counts, results, and the
SHA-256 of `data/splits.csv` (in SPEC.md). Whether a bare code list counts as
"data" under the terms is not certain; written confirmation from the
Manifesto Project would settle it.
