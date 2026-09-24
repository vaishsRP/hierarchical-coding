"""Sensitivity check for step 3 (see the dated note in SPEC.md): both cheap
fits chose C on the edge of the fixed grid, so extend each grid outward,
select on dev log-loss over the union, and score test for the choice.
The pre-registered numbers in results/cheap_baseline.json stay primary.

Writes results/cheap_baseline_sensitivity.json.
"""

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

import baseline_cheap as bc
import corpus

EXTRA = {"embeddings": [0.001, 0.003], "tfidf": [30.0, 100.0]}


def run(name, x_tr, y_tr, x_dev, y_dev, x_te, y_te, keep, parent):
    grid = sorted(set(bc.C_GRID) | set(EXTRA[name]))
    dev = {}
    best = None
    for c in grid:
        clf = LogisticRegression(C=c, max_iter=5000).fit(x_tr, y_tr)
        dev[c] = float(log_loss(y_dev, clf.predict_proba(x_dev), labels=clf.classes_))
        print(f"  {name} C={c:<7} dev log-loss {dev[c]:.4f}", flush=True)
        if best is None or dev[c] < dev[best[0]]:
            best = (c, clf)
    c, clf = best
    pred = clf.predict(x_te)
    return {"grid": grid, "dev_log_loss": dev, "chosen_C": c,
            "chosen_on_edge": c in (grid[0], grid[-1]),
            **bc.scores(list(y_te), list(pred), keep, parent)}


def main():
    units, keep, _, _, parent = bc.load()
    y = np.array([u["cmp_code"] for u in units])
    split = np.array([u["split"] for u in units])
    texts = [u["text"] or "" for u in units]
    tr, dev, te = (np.where(split == s)[0] for s in ("train", "dev", "test"))

    x = bc.embed(texts)
    sc = StandardScaler().fit(x[tr])
    xs = sc.transform(x)
    out = {"embeddings": run("embeddings", xs[tr], y[tr], xs[dev], y[dev], xs[te], y[te], keep, parent)}

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    xt = vec.fit_transform([texts[i] for i in tr])
    out["tfidf"] = run("tfidf", xt, y[tr], vec.transform([texts[i] for i in dev]), y[dev],
                       vec.transform([texts[i] for i in te]), y[te], keep, parent)

    (bc.RESULTS / corpus.res("cheap_baseline_sensitivity.json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in out.items():
        print(f"{k}: chosen C={v['chosen_C']} (edge: {v['chosen_on_edge']}), alpha {v['alpha_leaf']:.3f}, "
              f"macro F1 {v['macro_f1_leaf']:.3f}")


if __name__ == "__main__":
    main()
