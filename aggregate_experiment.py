"""Step 4: aggregate proportion experiment on the cheap baseline.

Question: when the per-item classifier gets better, does the error in the
per-document proportions (the number every Manifesto application reports)
go away, or does a systematic bias remain?

Protocol (fixed in SPEC.md before this was run):
  - unit of analysis: test manifesto d and modelled leaf k
  - true share p_dk = human-coded units of k in d / modelled units in d
  - predicted share q_dk, two estimators:
      classify-and-count (CC, headline): share of units whose argmax is k
      probability average (PA): mean predicted probability of k
  - signed error e_dk = q_dk - p_dk
  - per-leaf bias b_k = mean over test documents of e_dk, 95% CI from a
    bootstrap over test documents (2,000 resamples, seed 0)
  - summaries: mean absolute error over (d, k); total bias = sum_k |b_k| / 2
    (share of mass systematically put in the wrong leaf); Spearman rank
    correlation of b_k with the leaf's train prevalence
  - same summaries at category and domain level, shares derived through
    data/codeframe_hb5_structure.csv
  - learning curve: train fractions 5, 10, 25, 50, 100% of train units,
    sampled uniformly, 3 seeds below 100%; C re-chosen on dev log-loss each time

Writes results/aggregate_cheap.json and results/aggregate_cheap_per_leaf.csv.
"""

import csv
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler

import baseline_cheap as bc
import corpus

FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.00]
SEEDS = [0, 1, 2]
N_BOOT = 2000


def shares(doc_ids, labels_or_proba, classes, docs):
    """Rows: documents in `docs` order. Columns: `classes`.
    labels_or_proba is 1-D (labels, counted) or 2-D (probabilities, averaged)."""
    col = {c: j for j, c in enumerate(classes)}
    row = {d: i for i, d in enumerate(docs)}
    out = np.zeros((len(docs), len(classes)))
    n = np.zeros(len(docs))
    if labels_or_proba.ndim == 1:
        for d, lab in zip(doc_ids, labels_or_proba):
            out[row[d], col[lab]] += 1
            n[row[d]] += 1
    else:
        for d, p in zip(doc_ids, labels_or_proba):
            out[row[d]] += p
            n[row[d]] += 1
    return out / n[:, None]


def collapse(share, classes, level, parent):
    """Sum leaf shares into category or domain shares."""
    fn = corpus.category_of if level == "category" else corpus.domain_of
    groups = sorted({fn(c, parent) for c in classes})
    idx = {g: i for i, g in enumerate(groups)}
    out = np.zeros((share.shape[0], len(groups)))
    for j, c in enumerate(classes):
        out[:, idx[fn(c, parent)]] += share[:, j]
    return out, groups


def summarise(true, pred, prevalence, rng):
    err = pred - true
    bias = err.mean(axis=0)
    boot = np.empty((N_BOOT, err.shape[1]))
    for b in range(N_BOOT):
        boot[b] = err[rng.integers(0, err.shape[0], err.shape[0])].mean(axis=0)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    rho = spearmanr(bias, prevalence).statistic
    return {
        "mae": float(np.abs(err).mean()),
        "total_bias": float(np.abs(bias).sum() / 2),
        "spearman_bias_vs_prevalence": float(rho),
        "labels_with_ci_excluding_zero": int(((lo > 0) | (hi < 0)).sum()),
    }, bias, lo, hi


def aligned_proba(clf, x, keep):
    """Probabilities in `keep` column order. A leaf missing from a small
    train sample gets probability 1e-12 before renormalising."""
    proba = clf.predict_proba(x)
    full = np.full((x.shape[0], len(keep)), 1e-12)
    pos = {c: j for j, c in enumerate(keep)}
    for j, c in enumerate(clf.classes_):
        full[:, pos[c]] = np.maximum(proba[:, j], 1e-12)
    return full / full.sum(axis=1, keepdims=True)


def fit(x_tr, y_tr, x_dev, y_dev, keep):
    best = None
    for c in bc.C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000).fit(x_tr, y_tr)
        loss = log_loss(y_dev, aligned_proba(clf, x_dev, keep), labels=keep)
        if best is None or loss < best[0]:
            best = (loss, c, clf)
    return best


def evaluate(clf, xs_te, y_te, doc_te, docs, keep, parent, prevalence, detail=False):
    full = aligned_proba(clf, xs_te, keep)
    pred = np.array(keep)[full.argmax(axis=1)]

    true = shares(doc_te, y_te, keep, docs)
    est = {"cc": shares(doc_te, pred, keep, docs), "pa": shares(doc_te, full, keep, docs)}
    rng = np.random.default_rng(0)
    out = {"alpha_leaf": bc.alpha(list(y_te), list(pred)),
           "macro_f1_leaf": float(f1_score(y_te, pred, labels=keep, average="macro", zero_division=0))}
    rows = []
    for name, q in est.items():
        for level in ("leaf", "category", "domain"):
            if level == "leaf":
                t, p, groups, prev = true, q, keep, prevalence
            else:
                t, groups = collapse(true, keep, level, parent)
                p, _ = collapse(q, keep, level, parent)
                prev, _ = collapse(prevalence[None, :], keep, level, parent)
                prev = prev[0]
            summary, bias, lo, hi = summarise(t, p, prev, rng)
            out[f"{name}_{level}"] = summary
            if detail:
                rows += [[name, level, g, round(float(pv), 5), round(float(t[:, i].mean()), 5),
                          round(float(b), 5), round(float(l), 5), round(float(h), 5)]
                         for i, (g, pv, b, l, h) in enumerate(zip(groups, prev, bias, lo, hi))]
    return out, rows


def main():
    units, keep, sparse, counts, parent = bc.load()
    y = np.array([u["cmp_code"] for u in units])
    doc = np.array([u["manifesto_id"] for u in units])
    split = np.array([u["split"] for u in units])
    x = bc.embed([u["text"] or "" for u in units])
    tr, dev, te = (np.where(split == s)[0] for s in ("train", "dev", "test"))
    docs = sorted(set(doc[te]))

    prevalence = np.array([np.mean(y[tr] == c) for c in keep])
    curve = []
    detail_rows = None
    for frac in FRACTIONS:
        for seed in (SEEDS if frac < 1 else [0]):
            rng = np.random.default_rng(seed)
            sub = tr if frac == 1 else np.sort(rng.choice(tr, int(len(tr) * frac), replace=False))
            scaler = StandardScaler().fit(x[sub])
            loss, c, clf = fit(scaler.transform(x[sub]), y[sub], scaler.transform(x[dev]), y[dev], keep)
            res, rows = evaluate(clf, scaler.transform(x[te]), y[te], doc[te], docs,
                                 keep, parent, prevalence, detail=(frac == 1))
            res.update({"fraction": frac, "seed": seed, "train_units": int(len(sub)),
                        "C": c, "dev_log_loss": float(loss)})
            if frac == 1:
                detail_rows = rows
            curve.append(res)
            print(f"frac={frac:<5} seed={seed} C={c:<5} alpha={res['alpha_leaf']:.3f} "
                  f"macroF1={res['macro_f1_leaf']:.3f} CC leaf MAE={res['cc_leaf']['mae']:.4f} "
                  f"total bias={res['cc_leaf']['total_bias']:.3f} "
                  f"rho={res['cc_leaf']['spearman_bias_vs_prevalence']:.2f} | "
                  f"PA total bias={res['pa_leaf']['total_bias']:.3f}", flush=True)

    bc.RESULTS.mkdir(exist_ok=True)
    (bc.RESULTS / corpus.res("aggregate_cheap.json")).write_text(json.dumps(
        {"test_documents": len(docs), "curve": curve}, indent=2), encoding="utf-8")
    with (bc.RESULTS / corpus.res("aggregate_cheap_per_leaf.csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["estimator", "level", "code", "train_prevalence", "mean_true_share",
                    "bias", "ci_low", "ci_high"])
        w.writerows(detail_rows)
    print("wrote results/" + corpus.res("aggregate_cheap.json"))


if __name__ == "__main__":
    main()
