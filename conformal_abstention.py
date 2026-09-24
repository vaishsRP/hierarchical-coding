"""Step 7: conformal abstention, run here on the cheap baseline to build and
check the pipeline. Rerun on the strong model once it exists.

Protocol (fixed in SPEC.md before this was run):
  - model: the cheap baseline refit on train with the C chosen in step 3
  - split conformal with MAPIE, calibrated on the calib split, evaluated on
    test; target coverage 90%; conformity scores LAC and APS
  - an item is auto-coded if its prediction set has exactly one leaf;
    otherwise (empty or several leaves) it goes to a human
  - reported: empirical coverage on test, mean set size, human review load
    (share of items not auto-coded), accuracy of auto-coded items, and
    coverage per leaf and per country, because conformal only guarantees
    coverage on average and calib and test are different parties

Writes results/conformal_cheap.json.
"""

import collections
import json

import numpy as np
from mapie.classification import SplitConformalClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import baseline_cheap as bc
import corpus

TARGET = 0.90
SCORES = ["lac", "aps"]


def main():
    units, keep, _, _, _ = bc.load()
    y = np.array([u["cmp_code"] for u in units])
    split = np.array([u["split"] for u in units])
    country = np.array([u["country"] for u in units])
    x = bc.embed([u["text"] or "" for u in units])
    tr, cal, te = (np.where(split == s)[0] for s in ("train", "calib", "test"))

    c = json.loads((bc.RESULTS / corpus.res("cheap_baseline.json")).read_text())["cheap_baseline"]["C"]
    scaler = StandardScaler().fit(x[tr])
    xs = scaler.transform(x)
    clf = LogisticRegression(C=c, max_iter=3000).fit(xs[tr], y[tr])
    classes = list(clf.classes_)
    y_idx = np.array([classes.index(v) for v in y[te]])

    out = {"target_coverage": TARGET, "C": c, "calib_units": int(len(cal)),
           "test_units": int(len(te)), "plain_accuracy": float(np.mean(clf.predict(xs[te]) == y[te]))}
    for score in SCORES:
        mapie = SplitConformalClassifier(estimator=clf, confidence_level=TARGET,
                                         conformity_score=score, prefit=True, random_state=0)
        mapie.conformalize(xs[cal], y[cal])
        point, sets = mapie.predict_set(xs[te])
        sets = sets[:, :, 0]
        size = sets.sum(axis=1)
        covered = sets[np.arange(len(te)), y_idx]
        auto = size == 1
        per_leaf = {k: float(covered[y[te] == k].mean()) for k in keep}
        per_country = {k: float(covered[country[te] == k].mean()) for k in sorted(set(country[te]))}
        out[score] = {
            "coverage": float(covered.mean()),
            "mean_set_size": float(size.mean()),
            "empty_sets": float((size == 0).mean()),
            "human_review_load": float(1 - auto.mean()),
            "auto_coded_share": float(auto.mean()),
            "auto_coded_accuracy": float(np.mean(point[auto] == y[te][auto])) if auto.any() else None,
            "set_size_distribution": {int(k): int(v) for k, v in sorted(collections.Counter(size).items())[:10]},
            "leaves_below_80pct_coverage": sorted(k for k, v in per_leaf.items() if v < 0.8),
            "min_leaf_coverage": min(per_leaf.values()),
            "coverage_by_country": per_country,
        }
        r = out[score]
        print(f"{score}: coverage {r['coverage']:.3f}, mean set size {r['mean_set_size']:.2f}, "
              f"review load {r['human_review_load']:.3f}, auto-coded accuracy "
              f"{r['auto_coded_accuracy']:.3f}, leaves under 80% coverage "
              f"{len(r['leaves_below_80pct_coverage'])}", flush=True)

    (bc.RESULTS / corpus.res("conformal_cheap.json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/" + corpus.res("conformal_cheap.json"))


if __name__ == "__main__":
    main()
