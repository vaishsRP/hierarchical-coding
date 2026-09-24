"""Build the small files the Streamlit app reads, so the app itself needs no
corpus, no API key and no training.

  app/assets/model.npz            cheap baseline for the try-it box: scaler,
                                  logistic regression weights, and the split
                                  conformal (LAC, 90%) threshold from the calib split
  app/assets/category_names.json  readable names from the May 2021 handbook
  app/assets/summary.json         the numbers and chart data shown in the app

Contains no Manifesto text. The model and the names are derived from
Manifesto Project material; publishing them was a deliberate choice.
"""

import csv
import json
import math
import unicodedata

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import baseline_cheap as bc
import build_codeframe
import corpus

OUT = corpus.mp_api.ROOT / "app" / "assets"
TARGET = 0.90


def names():
    rows = build_codeframe.parse(build_codeframe.handbook_text("2021"))
    # PDF text carries ligatures (e.g. U+FB02 in "Influence"); NFKC unfolds them
    title = {r["code"]: unicodedata.normalize("NFKC", r["title"]) for r in rows}
    parent = {r["code"]: r["parent_code"] for r in rows}
    out = {}
    for code, t in title.items():
        p = parent[code]
        # a subcategory named only "General..." needs its category to make sense
        if "." in code and p in title and t.startswith("General"):
            t = f"{title[p]}: {t}"
        out[code] = t
    return out, {c: title[c] for c in title if c.startswith("domain_")}


def model():
    units, keep, _, _, _ = bc.load()
    y = np.array([u["cmp_code"] for u in units])
    split = np.array([u["split"] for u in units])
    x = bc.embed([u["text"] or "" for u in units])
    tr, cal = (np.where(split == s)[0] for s in ("train", "calib"))
    c = json.loads((bc.RESULTS / "cheap_baseline.json").read_text())["cheap_baseline"]["C"]
    scaler = StandardScaler().fit(x[tr])
    clf = LogisticRegression(C=c, max_iter=3000).fit(scaler.transform(x[tr]), y[tr])

    # split conformal, LAC score 1 - p(true label); finite-sample quantile
    p_cal = clf.predict_proba(scaler.transform(x[cal]))
    idx = {k: j for j, k in enumerate(clf.classes_)}
    scores = 1 - p_cal[np.arange(len(cal)), [idx[v] for v in y[cal]]]
    n = len(scores)
    q = float(np.quantile(scores, min(1.0, math.ceil((n + 1) * TARGET) / n), method="higher"))
    np.savez_compressed(OUT / "model.npz", classes=clf.classes_, coef=clf.coef_.astype(np.float32),
                        intercept=clf.intercept_.astype(np.float32),
                        mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32),
                        lac_threshold=np.float32(q))
    return q


# stance pairs: (the "against" side, its opposite)
STANCE_PAIRS = [("601.2", "602.2", "Against immigration"), ("505", "504", "Against expanding welfare"),
                ("110", "108", "Against the EU"), ("105", "104", "Against the military")]


def stance_flips():
    """How often the cheap model tags a sentence with the opposite stance."""
    z = np.load(corpus.mp_api.CACHE_DIR / "preds_cheap_test.npz", allow_pickle=True)
    classes = list(z["classes"])
    y = z["y_true"]
    pred = np.array(classes)[z["proba"].argmax(axis=1)]
    out = []
    for code, other, label in STANCE_PAIRS:
        m = y == code
        out.append({"stance": label, "sentences": int(m.sum()),
                    "correct": float(np.mean(pred[m] == code)), "flipped": float(np.mean(pred[m] == other))})
    return out


BES_ISSUES = ["Europe", "Immigration", "Economy-general", "Health"]


def bes_block():
    """Reported share of the biggest issues per BES wave, and the switch estimate.
    Aggregate shares only; wave dates are the median interview date."""
    import pandas as pd
    out = json.loads((bc.RESULTS / "bes_event_study.json").read_text())
    codes = pd.read_pickle(corpus.mp_api.CACHE_DIR / "bes" / "mii_codes.pkl")
    when = {}
    for w in range(1, 31):
        t = pd.to_datetime(codes.get(f"starttimeW{w}"), errors="coerce").dropna()
        if len(t):
            when[w] = str(t.median().date())
    series = []
    with (bc.RESULTS / "bes_gaps_by_wave.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["name"] in BES_ISSUES and int(r["wave"]) in when:
                series.append({"issue": r["name"].replace("-general", ""), "wave": int(r["wave"]),
                               "date": when[int(r["wave"])], "share": float(r["bes_share"]),
                               "coder": r["coder"]})
    placebo = out["placebo"]["total_shift"]
    return {"series": series, "switch_date": when[26], "shift": out["total_shift"],
            "drift_median": float(np.median(placebo)), "drift_max": float(max(placebo)),
            "respondents": out["respondents"]}


def summary(leaf_names, domain_names):
    cheap = json.loads((bc.RESULTS / "cheap_baseline.json").read_text())
    agg = json.loads((bc.RESULTS / "aggregate_cheap.json").read_text())
    ppi = json.loads((bc.RESULTS / "ppi_cheap.json").read_text())
    conf = json.loads((bc.RESULTS / "conformal_cheap.json").read_text())

    curve = {}
    for r in agg["curve"]:
        c = curve.setdefault(r["train_units"], [])
        c.append(r)
    learning = [{"train_sentences": int(k),
                 "count_top_guess": float(np.mean([r["cc_leaf"]["total_bias"] for r in v])),
                 "average_probabilities": float(np.mean([r["pa_leaf"]["total_bias"] for r in v])),
                 "accuracy_proxy_alpha": float(np.mean([r["alpha_leaf"] for r in v]))}
                for k, v in sorted(curve.items())]

    bias = []
    with (bc.RESULTS / "aggregate_cheap_per_leaf.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["estimator"] == "cc" and r["level"] == "leaf":
                bias.append({"code": r["code"], "name": leaf_names.get(r["code"], r["code"]),
                             "bias_points": 100 * float(r["bias"]),
                             "ci_low": 100 * float(r["ci_low"]), "ci_high": 100 * float(r["ci_high"]),
                             "true_share": 100 * float(r["mean_true_share"])})
    bias.sort(key=lambda b: -abs(b["bias_points"]))

    return {
        "model": "sentence embeddings + logistic regression (cheap baseline)",
        "test": {"manifestos": agg["test_documents"], "sentences": cheap["data"]["split_units"]["test"],
                 "categories": cheap["data"]["leaves"]},
        "item": {"accuracy": cheap["cheap_baseline"]["accuracy_leaf"],
                 "alpha": cheap["cheap_baseline"]["alpha_leaf"],
                 "domain_accuracy": cheap["cheap_baseline"]["accuracy_domain"]},
        "learning_curve": learning,
        "category_bias": bias,
        "ppi": {b: {"model_only": v["cc"]["total_bias"], "with_checks": v["ppi"]["total_bias"],
                    "labels_only": v["labels"]["total_bias"],
                    "interval_hit_rate": v["ppi"]["ci95_coverage_share_ge_2pct"]}
                for b, v in ppi["budgets"].items()},
        "review": {"target": conf["target_coverage"], "review_load": conf["lac"]["human_review_load"],
                   "auto_accuracy": conf["lac"]["auto_coded_accuracy"], "coverage": conf["lac"]["coverage"]},
        "domains": domain_names,
        "stance_flips": stance_flips(),
        "bes": bes_block(),
        "llm_wrong_topic": json.loads((bc.RESULTS / "llm_zero_shot.json").read_text())
                           ["samples"]["random"]["llm"]["pooled_wrong_category"],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    leaf_names, domain_names = names()
    (OUT / "category_names.json").write_text(json.dumps(leaf_names, indent=1, ensure_ascii=False), encoding="utf-8")
    q = model()
    (OUT / "summary.json").write_text(json.dumps(summary(leaf_names, domain_names), indent=1,
                                                 ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}; LAC threshold {q:.4f}")


if __name__ == "__main__":
    main()
