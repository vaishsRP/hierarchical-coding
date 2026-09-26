"""Build data/raw/bes/mii_codes.pkl, the BES "most important issue" codes per
wave, from the BES combined panel file. bes_event_study.py reads it.

Download from britishelectionstudy.com (Data, Panel study data; free after
registration, BES terms apply):
  the Combined Wave 1-31 Internet Panel (Stata)
  the Combined Wave 1-30 open ended response data (Stata)
Point BES_DIR at the folder that holds them (default: your Downloads folder).

Only codes, respondent ids and interview dates are extracted; the answer text
is read later by bes_event_study.py. Everything stays in data/raw/ (gitignored).
"""

import os
import re
from pathlib import Path

import pandas as pd

import corpus

BES_DIR = Path(os.environ.get("BES_DIR", Path.home() / "Downloads"))
PANEL = BES_DIR / "BES2024_W31_Panel_v31.05.dta"
OUT = corpus.mp_api.CACHE_DIR / "bes" / "mii_codes.pkl"


def main():
    r = pd.io.stata.StataReader(PANEL)
    labels = r.variable_labels()
    cols = ["id"] + [k for k in labels if re.fullmatch(r"(small_)?mii_cat(_llm)?W\d+", k)]
    cols += [k for k in labels if re.fullmatch(r"starttimeW\d+", k)]
    codes = pd.read_stata(PANEL, columns=cols, convert_categoricals=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    codes.to_pickle(OUT)
    print(f"wrote {len(codes):,} respondents and {len(cols) - 1} columns to {OUT}")


if __name__ == "__main__":
    main()
