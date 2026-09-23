"""Step 8: prediction-powered inference (PPI) for per-document shares, run
here on the cheap baseline. Rerun on the strong model once it exists.

For document d and leaf k, with N units, model indicator f_i = [argmax is k]
and a uniform random sample of n units whose human codes y_i are known:

  predictions only (CC):  mean over all N of f_i                (no CI)
  labels only:            mean over n of y_i                    CI: CLT
  PPI:                    mean over N of f_i - mean over n of (f_i - y_i)
                          CI: z * sqrt(var(f)/N + var(f - y)/n)

PPI is unbiased for the true share whatever the model's errors are; the
model only decides how narrow the interval is.

Protocol (fixed in SPEC.md before this was run):
  - model: the cheap baseline refit on train with the step 3 C, test documents
  - label budgets: 5, 10 and 20% of each document's units, at least 10 units
  - 200 random label samples per budget, seeds 0 to 199
  - per estimator: total bias (sum over leaves of |mean signed error| / 2,
    averaged over documents and draws), RMSE, and for the two estimators
    with intervals, 95% CI coverage of the true share and mean CI width.
    Coverage is reported over all (document, leaf) pairs and over pairs with
    a true share of at least 2%: for rare leaves a small sample often holds
    no example, the CLT interval collapses to zero width, and all-pairs
    coverage then mixes trivially correct zeros with real misses
  - shares are over modelled units only, as in step 4

Writes results/ppi_cheap.json.
"""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import baseline_cheap as bc

BUDGETS = [0.05, 0.10, 0.20]
MIN_LABELS = 10
DRAWS = 200
Z = 1.959964


def main():
    units, keep, _, _, _ = bc.load()
    y = np.array([u["cmp_code"] for u in units])
    doc = np.array([u["manifesto_id"] for u in units])
    split = np.array([u["split"] for u in units])
    x = bc.embed([u["text"] or "" for u in units])
    tr, te = (np.where(split == s)[0] for s in ("train", "test"))

    c = json.loads((bc.RESULTS / "cheap_baseline.json").read_text())["cheap_baseline"]["C"]
    scaler = StandardScaler().fit(x[tr])
    clf = LogisticRegression(C=c, max_iter=3000).fit(scaler.transform(x[tr]), y[tr])
    pred = clf.predict(scaler.transform(x[te]))

    col = {k: j for j, k in enumerate(keep)}
    Y = np.zeros((len(te), len(keep)))
    F = np.zeros((len(te), len(keep)))
    Y[np.arange(len(te)), [col[v] for v in y[te]]] = 1
    F[np.arange(len(te)), [col[v] for v in pred]] = 1
    docs = sorted(set(doc[te]))
    rows_of = {d: np.where(doc[te] == d)[0] for d in docs}

    out = {"test_documents": len(docs), "draws": DRAWS, "budgets": {}}
    for budget in BUDGETS:
        err = {"cc": [], "labels": [], "ppi": []}
        cover = {"labels": [], "ppi": []}
        big = []
        width = {"labels": [], "ppi": []}
        for d in docs:
            r = rows_of[d]
            yd, fd = Y[r], F[r]
            truth = yd.mean(axis=0)
            n_all = len(r)
            n = min(n_all, max(MIN_LABELS, int(round(budget * n_all))))
            cc = fd.mean(axis=0)
            var_f = fd.var(axis=0, ddof=1)
            big_d = truth >= 0.02
            for seed in range(DRAWS):
                big.append(big_d)
                s = np.random.default_rng(seed).choice(n_all, n, replace=False)
                lab = yd[s].mean(axis=0)
                rect = (fd[s] - yd[s])
                ppi = cc - rect.mean(axis=0)
                half_lab = Z * np.sqrt(yd[s].var(axis=0, ddof=1) / n)
                half_ppi = Z * np.sqrt(var_f / n_all + rect.var(axis=0, ddof=1) / n)
                err["cc"].append(cc - truth)
                err["labels"].append(lab - truth)
                err["ppi"].append(ppi - truth)
                for name, est, half in (("labels", lab, half_lab), ("ppi", ppi, half_ppi)):
                    cover[name].append(np.abs(est - truth) <= half)
                    width[name].append(2 * half)
        res = {}
        for name, e in err.items():
            e = np.array(e)
            res[name] = {"total_bias": float(np.abs(e.mean(axis=0)).sum() / 2),
                         "rmse": float(np.sqrt((e ** 2).mean()))}
            if name in cover:
                cv = np.array(cover[name])
                res[name]["ci95_coverage"] = float(cv.mean())
                res[name]["ci95_coverage_share_ge_2pct"] = float(cv[np.array(big)].mean())
                res[name]["mean_ci_width"] = float(np.mean(width[name]))
        out["budgets"][str(budget)] = res
        print(f"budget {budget:.0%}: " + " | ".join(
            f"{k} bias={v['total_bias']:.4f} rmse={v['rmse']:.4f}"
            + (f" cov={v['ci95_coverage']:.3f}/{v['ci95_coverage_share_ge_2pct']:.3f} width={v['mean_ci_width']:.4f}" if "ci95_coverage" in v else "")
            for k, v in res.items()), flush=True)

    (bc.RESULTS / "ppi_cheap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/ppi_cheap.json")


if __name__ == "__main__":
    main()
