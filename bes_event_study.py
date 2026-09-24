"""BES: what did the switch from human to LLM coding do to the reported
shares of "most important issue"? Protocol fixed in SPEC.md ("BES protocol").

A model trained on human coded waves 1 to 25 is one consistent coder across
all 31 waves. Before the switch, the gap between BES's shares and the model's
shares is the model's own bias; after the switch it is that bias plus the
effect of LLM coding. The change in the gap estimates the switch.

Everything runs locally (BES terms: no third parties). Inputs, all gitignored:
  data/raw/bes/mii_codes.pkl                 codes per wave (from the panel file)
  Downloads/BES2024_W30Strings_v30.1.dta     text, waves 1 to 30
  Downloads/BES2024_W31_v31.05.dta           text, wave 31
Outputs (aggregate shares only, no text):
  results/bes_event_study.json, results/bes_gaps_by_wave.csv
"""

import codecs
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

import baseline_cheap as bc
import corpus

DOWNLOADS = Path.home() / "Downloads"
PANEL = DOWNLOADS / "BES2024_W31_Panel_v31.05.dta"
STRINGS = DOWNLOADS / "BES2024_W30Strings_v30.1.dta"
W31 = DOWNLOADS / "BES2024_W31_v31.05.dta"
CODES = corpus.mp_api.CACHE_DIR / "bes" / "mii_codes.pkl"
LONG = corpus.mp_api.CACHE_DIR / "bes" / "mii_long.pkl"
SEED = 20260924
HUMAN, LLM = range(1, 26), range(26, 32)
PRE, POST = range(20, 26), range(26, 32)
SLOPE = range(14, 26)
C_GRID = [0.01, 0.1, 1.0]
BOOT = 1000


def _decode_mixed(data, errors="strict"):
    """The strings file mixes UTF-8 and Windows-1252 text (e.g. byte 0x92 for a
    curly apostrophe). Try UTF-8 first, then Windows-1252, then Latin-1."""
    data = bytes(data)
    for enc in ("utf-8", "cp1252"):
        try:
            return data.decode(enc), len(data)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1"), len(data)


codecs.register(lambda name: codecs.CodecInfo(codecs.utf_8_encode, _decode_mixed, name="utf8_mixed")
                if name == "utf8_mixed" else None)


def read_text_columns(path, cols):
    reader = pd.io.stata.StataReader(path, columns=cols, convert_categoricals=False, chunksize=20000)
    with reader:
        reader.variable_labels()          # parse the header first: it resets the encoding
        reader._encoding = "utf8_mixed"
        return pd.concat(list(reader), ignore_index=True)


def long_table():
    """One row per (respondent, wave) with text and BES code."""
    if LONG.exists():
        return pd.read_pickle(LONG)
    codes = pd.read_pickle(CODES)
    text_cols = [f"MII_textW{w}" for w in range(1, 31)]
    # the strings file is 2 GB: read it in chunks, keeping only the MII text columns
    txt = read_text_columns(STRINGS, ["id"] + text_cols)
    w31 = pd.read_stata(W31, columns=["id", "mii"], convert_categoricals=True)
    txt = txt.merge(w31.rename(columns={"mii": "MII_textW31"}), on="id", how="outer")
    rows = []
    for w in range(1, 32):
        code_col = f"mii_catW{w}" if w in HUMAN else f"mii_cat_llmW{w}"
        part = codes[["id", code_col]].merge(txt[["id", f"MII_textW{w}"]], on="id")
        part.columns = ["id", "code", "text"]
        part["text"] = part["text"].astype("string").str.strip()
        part = part.dropna(subset=["code", "text"])
        part = part[part["text"].str.len() > 0]
        part["wave"] = w
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    out["code"] = out["code"].astype(int)
    out.to_pickle(LONG)
    return out


def split(ids):
    """70% train, 30% evaluation by respondent; 10% of train respondents are dev."""
    def u(i, salt):
        return int(hashlib.sha256(f"{SEED}:{salt}:{i}".encode()).hexdigest()[:8], 16) / 16 ** 8
    return {i: ("eval" if u(i, "e") < 0.30 else "dev" if u(i, "d") < 0.10 else "train") for i in ids}


def category_names():
    r = pd.io.stata.StataReader(PANEL)
    r.variable_labels()
    lbl = dict(zip(r._varlist, r._lbllist))
    return {int(k): v for k, v in r.value_labels()[lbl["mii_catW20"]].items()}


def shares_by_wave(df, probs, classes):
    """BES share and model share (averaged probabilities, and top guess) per wave."""
    col = {c: j for j, c in enumerate(classes)}
    out = {}
    for w, g in df.groupby("wave"):
        idx = g.index.to_numpy()
        onehot = np.zeros((len(g), len(classes)))
        onehot[np.arange(len(g)), [col[c] for c in g["code"]]] = 1
        top = np.zeros_like(onehot)
        top[np.arange(len(g)), probs[idx].argmax(axis=1)] = 1
        out[w] = (onehot, probs[idx], top, g["id"].to_numpy())
    return out


def estimate(per_wave, weights=None):
    """gap(w, k) and the post minus pre shift, optionally with respondent weights."""
    gaps, gaps_cc = {}, {}
    for w, (onehot, pa, top, ids) in per_wave.items():
        wt = np.ones(len(ids)) if weights is None else weights[ids]
        s = wt.sum()
        bes, mod, cc = (wt @ onehot) / s, (wt @ pa) / s, (wt @ top) / s
        gaps[w], gaps_cc[w] = bes - mod, bes - cc
    shift = np.mean([gaps[w] for w in POST], axis=0) - np.mean([gaps[w] for w in PRE], axis=0)
    shift_cc = np.mean([gaps_cc[w] for w in POST], axis=0) - np.mean([gaps_cc[w] for w in PRE], axis=0)
    return gaps, shift, shift_cc


def main():
    df = long_table()
    names = category_names()
    print(f"{len(df):,} coded answers, waves {df.wave.min()} to {df.wave.max()}, "
          f"{df.id.nunique():,} respondents, {df.text.nunique():,} distinct texts", flush=True)
    part = split(df["id"].unique())
    df["part"] = df["id"].map(part)

    uniq = pd.Series(df["text"].unique())
    emb = bc.embed(list(uniq))                       # cached; distinct texts only
    row = {t: i for i, t in enumerate(uniq)}
    x = emb[df["text"].map(row).to_numpy()]

    human = df["wave"].isin(HUMAN)
    tr = np.where(human & (df["part"] == "train"))[0]
    dv = np.where(human & (df["part"] == "dev"))[0]
    scaler = StandardScaler().fit(x[tr])
    xs_tr, xs_dv = scaler.transform(x[tr]), scaler.transform(x[dv])
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000).fit(xs_tr, df["code"].to_numpy()[tr])
        loss = log_loss(df["code"].to_numpy()[dv], clf.predict_proba(xs_dv), labels=clf.classes_)
        print(f"  C={c:<5} dev log-loss {loss:.4f}", flush=True)
        if best is None or loss < best[0]:
            best = (loss, c, clf)
    loss, c, clf = best
    classes = list(clf.classes_)

    ev = df[df["part"] == "eval"].copy()
    ev = ev[ev["code"].isin(classes)]
    probs = np.zeros((len(df), len(classes)))
    probs[ev.index.to_numpy()] = clf.predict_proba(scaler.transform(x[ev.index.to_numpy()]))
    per_wave = shares_by_wave(ev, probs, classes)
    gaps, shift, shift_cc = estimate(per_wave)

    # bootstrap over evaluation respondents (Poisson weights)
    ids = ev["id"].unique()
    pos = {i: j for j, i in enumerate(ids)}
    per_wave_idx = {w: (a, b, t, np.array([pos[i] for i in r])) for w, (a, b, t, r) in per_wave.items()}
    rng = np.random.default_rng(SEED)
    boot = np.array([estimate(per_wave_idx, rng.poisson(1.0, len(ids)).astype(float))[1] for _ in range(BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    total = np.abs(boot).sum(axis=1) / 2

    waves = sorted(gaps)
    slope = {k: float(np.polyfit(list(SLOPE), [gaps[w][j] for w in SLOPE], 1)[0])
             for j, k in enumerate(classes)}
    cats = sorted(({"code": int(k), "name": names.get(int(k), str(k)), "shift_points": 100 * float(shift[j]),
                    "ci_low": 100 * float(lo[j]), "ci_high": 100 * float(hi[j]),
                    "shift_points_top_guess": 100 * float(shift_cc[j]),
                    "pre_slope_points_per_wave": 100 * slope[k],
                    "bes_share_pre": 100 * float(np.mean([per_wave[w][0].mean(0)[j] for w in PRE])),
                    "bes_share_post": 100 * float(np.mean([per_wave[w][0].mean(0)[j] for w in POST]))}
                   for j, k in enumerate(classes)), key=lambda r: -abs(r["shift_points"]))
    out = {"answers": int(len(df)), "eval_answers": int(len(ev)), "respondents": int(df.id.nunique()),
           "C": c, "dev_log_loss": float(loss),
           "total_shift": float(np.abs(shift).sum() / 2),
           "total_shift_ci": [float(np.percentile(total, 2.5)), float(np.percentile(total, 97.5))],
           "categories_ci_excluding_zero": int(((lo > 0) | (hi < 0)).sum()),
           "categories": cats}
    bc.RESULTS.mkdir(exist_ok=True)
    (bc.RESULTS / "bes_event_study.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    with (bc.RESULTS / "bes_gaps_by_wave.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["wave", "coder", "code", "name", "bes_share", "model_share", "gap"])
        for w in waves:
            onehot, pa, _, _ = per_wave[w]
            for j, k in enumerate(classes):
                wr.writerow([w, "human" if w in HUMAN else "llm", int(k), names.get(int(k), k),
                             round(float(onehot[:, j].mean()), 5), round(float(pa[:, j].mean()), 5),
                             round(float(gaps[w][j]), 5)])
    print(f"total shift {out['total_shift']:.3f} (95% CI {out['total_shift_ci'][0]:.3f} to "
          f"{out['total_shift_ci'][1]:.3f}); {out['categories_ci_excluding_zero']} categories significant")
    for r in cats[:8]:
        print(f"  {r['name'][:40]:40} {r['shift_points']:+.2f} points [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
              f"  pre slope {r['pre_slope_points_per_wave']:+.3f}")


if __name__ == "__main__":
    main()
