"""Paired bootstrap over test manifestos for the main model comparisons
(added after review, 2026-09-26). Resamples the 24 English test manifestos with
replacement and recomputes, for each model, alpha, macro F1 and total bias
(the systematic share of content in the wrong topic, counting top guesses).
Reports each difference with a 95% interval.

Writes results/bootstrap_compare.json.
"""

import json

import numpy as np

import baseline_cheap as bc
import corpus

B = 2000
PAIRS = [("deberta_flat", "cheap"), ("deberta_definitions", "deberta_flat")]


def gold():
    rows = [json.loads(l) for l in corpus.MODELLING.open(encoding="utf-8")]
    return {f"{r['manifesto_id']}#{r['pos']}": r for r in rows if r["split"] == "test"}


def predictions(ids, keep):
    """Top guess per test unit for each model (one array per seed)."""
    col = {k: j for j, k in enumerate(keep)}
    out = {}
    # read each array once: indexing an npz file inside the loop re-reads it every time
    z = np.load(corpus.mp_api.CACHE_DIR / "preds_cheap_test.npz", allow_pickle=True)
    top = z["proba"].argmax(axis=1)
    classes = list(z["classes"])
    pos = {f"{m}#{p}": i for i, (m, p) in enumerate(zip(z["manifesto_id"], z["pos"]))}
    out["cheap"] = [np.array([col[classes[top[pos[u]]]] for u in ids])]
    for name, pattern in (("deberta_flat", "runs/kaggle_full/runs/deberta_s{}/probs.npz"),
                          ("deberta_definitions", "runs/kaggle_hc-structured-defs/runs/defs_s{}/probs.npz")):
        seeds = []
        for s in range(3):
            z = np.load(corpus.mp_api.ROOT / pattern.format(s))
            top = z["test_proba"].argmax(axis=1)
            labels = list(z["labels"])
            pos = {u: i for i, u in enumerate(z["test_ids"])}
            seeds.append(np.array([col[labels[top[pos[u]]]] for u in ids]))
        out[name] = seeds
    return out


def per_document(y, pred, doc, docs, k):
    """Confusion matrix and true/predicted shares for each document."""
    conf = np.zeros((len(docs), k, k))
    true_share = np.zeros((len(docs), k))
    pred_share = np.zeros((len(docs), k))
    for i, d in enumerate(docs):
        m = doc == d
        np.add.at(conf[i], (y[m], pred[m]), 1)
        true_share[i] = np.bincount(y[m], minlength=k) / m.sum()
        pred_share[i] = np.bincount(pred[m], minlength=k) / m.sum()
    return conf, true_share, pred_share


def metrics(conf, true_share, pred_share, w):
    c = np.tensordot(w, conf, axes=1)                      # weighted total confusion
    o = c + c.T                                            # coincidence matrix, two coders
    n_c = o.sum(axis=1)
    n = n_c.sum()
    alpha = 1 - ((n - np.trace(o)) / n) / ((n * n - (n_c ** 2).sum()) / (n * (n - 1)))
    tp = np.diag(c)
    prec = np.divide(tp, c.sum(axis=0), out=np.zeros_like(tp), where=c.sum(axis=0) > 0)
    rec = np.divide(tp, c.sum(axis=1), out=np.zeros_like(tp), where=c.sum(axis=1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    bias = np.abs((w[:, None] * (pred_share - true_share)).sum(0) / w.sum()).sum() / 2
    return np.array([alpha, f1.mean(), bias])


def main():
    units, keep, _, _, _ = bc.load()
    g = gold()
    ids = sorted(g)
    y = np.array([keep.index(g[u]["label"]) for u in ids])
    doc = np.array([g[u]["manifesto_id"] for u in ids])
    docs = sorted(set(doc))
    k = len(keep)
    per = {name: [per_document(y, p, doc, docs, k) for p in seeds]
           for name, seeds in predictions(ids, keep).items()}

    def score(name, w):
        return np.mean([metrics(*parts, w) for parts in per[name]], axis=0)

    rng = np.random.default_rng(0)
    ones = np.ones(len(docs))
    out = {"documents": len(docs), "resamples": B, "comparisons": {}}
    for a, b in PAIRS:
        diffs = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, len(docs), len(docs)), minlength=len(docs)).astype(float)
            diffs.append(score(a, w) - score(b, w))
        diffs = np.array(diffs)
        point = score(a, ones) - score(b, ones)
        lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
        out["comparisons"][f"{a} minus {b}"] = {
            m: {"difference": float(point[i]), "ci_low": float(lo[i]), "ci_high": float(hi[i])}
            for i, m in enumerate(("alpha", "macro_f1", "total_bias"))}
    (bc.RESULTS / "bootstrap_compare.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, v in out["comparisons"].items():
        print(name)
        for m, r in v.items():
            print(f"  {m:11} {r['difference']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
