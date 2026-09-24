"""Score fine tuned runs (steps 5 and 6) from their probs.npz files with the
same measures as steps 3, 4, 7 and 8, and apply the step 6 decision rule.

For each run: per sentence scores (and the strict domain first decode when
domain probabilities are present), aggregate bias at full training data
(classify and count, probability average), split conformal review load (LAC
and APS, 90%, calibrated on calib), and PPI at 5, 10 and 20%.

Conformal sets are computed here directly from the saved probabilities, so
MAPIE is not needed. `--check` recomputes the cheap baseline's sets this way
and compares them with results/conformal_cheap.json (MAPIE): LAC matches
MAPIE exactly. APS here is the non randomised version, which over covers
(0.949 against MAPIE's randomised 0.914 on the cheap model), so it is
labelled aps_nonrandom and LAC stays the headline.

`--posthoc` first writes <run>/posthoc/probs.npz for every run: temperature
scaling (T by dev log loss) then logit adjustment for class frequency (tau by
dev macro F1), both fitted per run on dev only (see the dated note in SPEC.md),
and scores those instead.

Usage:
  python evaluate_probs.py --check
  python evaluate_probs.py --name flat runs/kaggle_full/runs/deberta_s0 ...
  python evaluate_probs.py --compare flat hier defs
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

import aggregate_experiment as ae
import baseline_cheap as bc
import corpus
import ppi_aggregate as pa

TARGET = 0.90
KEY_METRICS = ["alpha_leaf", "macro_f1_leaf", "accuracy_leaf", "alpha_domain",
               "cc_total_bias", "pa_total_bias", "review_load_lac", "auto_accuracy_lac"]


def gold():
    rows = [json.loads(l) for l in (corpus.MODELLING).open(encoding="utf-8")]
    return {f"{r['manifesto_id']}#{r['pos']}": r for r in rows}


def quantile(scores, target=TARGET):
    n = len(scores)
    return float(np.quantile(scores, min(1.0, math.ceil((n + 1) * target) / n), method="higher"))


def conformal(p_cal, y_cal, p_te, y_te):
    """y_* are column indices. Returns metrics for LAC and non randomised APS."""
    out = {}
    # LAC: score 1 - p(true); set = labels with p >= 1 - q
    q = quantile(1 - p_cal[np.arange(len(y_cal)), y_cal])
    sets = {"lac": p_te >= 1 - q}
    # APS: score = total probability of labels ranked at or above the true one
    def aps_scores(p, y):
        order = np.argsort(-p, axis=1)
        cum = np.cumsum(np.take_along_axis(p, order, axis=1), axis=1)
        rank = np.argsort(order, axis=1)[np.arange(len(y)), y]
        return cum[np.arange(len(y)), rank]
    q_aps = quantile(aps_scores(p_cal, y_cal))
    order = np.argsort(-p_te, axis=1)
    cum = np.cumsum(np.take_along_axis(p_te, order, axis=1), axis=1)
    keep_sorted = (cum - np.take_along_axis(p_te, order, axis=1)) < q_aps     # include until mass reaches q
    aps = np.zeros_like(p_te, dtype=bool)
    np.put_along_axis(aps, order, keep_sorted, axis=1)
    sets["aps_nonrandom"] = aps
    top = p_te.argmax(axis=1)
    for name, s in sets.items():
        size = s.sum(axis=1)
        covered = s[np.arange(len(y_te)), y_te]
        auto = size == 1
        out[name] = {"coverage": float(covered.mean()), "mean_set_size": float(size.mean()),
                     "human_review_load": float(1 - auto.mean()),
                     "auto_accuracy": float((top[auto] == y_te[auto]).mean()) if auto.any() else None}
    return out


def score_run(run, g, keep, parent, prevalence):
    z = np.load(Path(run) / "probs.npz")
    labels = list(z["labels"])
    assert labels == keep, "label order differs from the modelling set"
    col = {k: j for j, k in enumerate(keep)}
    res = {}

    te = [g[i] for i in z["test_ids"]]
    y_te = np.array([r["label"] for r in te])
    p_te = z["test_proba"]
    pred = np.array(keep)[p_te.argmax(axis=1)]
    res.update(bc.scores(list(y_te), list(pred), keep, parent))
    res["test_log_loss"] = float(log_loss(y_te, p_te, labels=keep))

    if "test_domain_proba" in z.files:
        domains = list(z["domains"])
        leaf_dom = np.array([domains.index(corpus.domain_of(c, parent)) for c in keep])
        d_hat = z["test_domain_proba"].argmax(axis=1)
        masked = np.where(leaf_dom[None, :] == d_hat[:, None], p_te, -1.0)
        strict = np.array(keep)[masked.argmax(axis=1)]
        s = bc.scores(list(y_te), list(strict), keep, parent)
        res["strict_domain_first"] = {k: s[k] for k in ("alpha_leaf", "macro_f1_leaf", "accuracy_leaf", "alpha_domain")}

    doc = np.array([r["manifesto_id"] for r in te])
    docs = sorted(set(doc))
    true = ae.shares(doc, y_te, keep, docs)
    rng = np.random.default_rng(0)
    for name, q in (("cc", ae.shares(doc, pred, keep, docs)), ("pa", ae.shares(doc, p_te, keep, docs))):
        summary, _, _, _ = ae.summarise(true, q, prevalence, rng)
        res[f"{name}_total_bias"] = summary["total_bias"]
        res[f"{name}_leaf"] = summary

    cal = [g[i] for i in z["calib_ids"]]
    conf = conformal(z["calib_proba"], np.array([col[r["label"]] for r in cal]),
                     p_te, np.array([col[v] for v in y_te]))
    res["conformal"] = conf
    res["review_load_lac"] = conf["lac"]["human_review_load"]
    res["auto_accuracy_lac"] = conf["lac"]["auto_accuracy"]

    Y, F = pa.indicators(y_te, pred, keep)
    res["ppi"] = pa.ppi_budgets(Y, F, doc, verbose=False)
    res["best_epoch"] = json.loads((Path(run) / "log.json").read_text()).get("best_epoch")
    return res


TAUS = [0.0, 0.25, 0.5, 0.75, 1.0]


def softmax(a):
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def posthoc(run, g, keep, prior, taus=None):
    """Temperature scaling then logit adjustment, fitted on dev only."""
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import f1_score
    z = np.load(Path(run) / "probs.npz")
    col = {k: j for j, k in enumerate(keep)}
    y_dev = np.array([col[g[i]["label"]] for i in z["dev_ids"]])
    logp = {s: np.log(np.clip(z[f"{s}_proba"], 1e-12, 1)) for s in ("dev", "calib", "test")}

    def nll(t):
        q = softmax(logp["dev"] / t)
        return -np.log(np.clip(q[np.arange(len(y_dev)), y_dev], 1e-12, 1)).mean()
    t = float(minimize_scalar(nll, bounds=(0.25, 10), method="bounded").x)

    log_prior = np.log(prior)
    scores = {}
    taus = taus or TAUS
    for tau in taus:
        pred = (logp["dev"] / t - tau * log_prior).argmax(axis=1)
        scores[tau] = f1_score(y_dev, pred, labels=list(range(len(keep))), average="macro", zero_division=0)
    tau = max(taus, key=lambda k: (round(scores[k], 6), -k))

    out = Path(run) / ("posthoc" if len(taus) > 1 else "temperature")
    out.mkdir(exist_ok=True)
    arrays = {"labels": z["labels"]}
    for s in ("dev", "calib", "test"):
        arrays[f"{s}_proba"] = softmax(logp[s] / t - tau * log_prior)
        arrays[f"{s}_ids"] = z[f"{s}_ids"]
    np.savez_compressed(out / "probs.npz", **arrays)
    log = json.loads((Path(run) / "log.json").read_text())
    log["posthoc"] = {"temperature": t, "tau": tau, "dev_macro_f1_by_tau": scores}
    (out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"{run}: T={t:.3f} tau={tau} dev macro F1 {scores[tau]:.4f} (tau 0: {scores[0.0]:.4f})", flush=True)
    return str(out)


def summarise_runs(runs):
    out = {}
    for m in KEY_METRICS:
        v = [r[m] for r in runs.values() if r.get(m) is not None]
        out[m] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    if all("strict_domain_first" in r for r in runs.values()):
        for m in ("alpha_leaf", "macro_f1_leaf"):
            v = [r["strict_domain_first"][m] for r in runs.values()]
            out[f"strict_{m}"] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    for b in pa.BUDGETS:
        for est in ("labels", "ppi"):
            for k in ("rmse", "ci95_coverage_share_ge_2pct", "mean_ci_width"):
                v = [r["ppi"][str(b)][est][k] for r in runs.values()]
                out[f"ppi_{b}_{est}_{k}"] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
    return out


def check():
    """Our conformal code against MAPIE's numbers for the cheap baseline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    units, keep, _, _, _ = bc.load()
    y = np.array([u["cmp_code"] for u in units])
    split = np.array([u["split"] for u in units])
    x = bc.embed([u["text"] or "" for u in units])
    tr, cal, te = (np.where(split == s)[0] for s in ("train", "calib", "test"))
    c = json.loads((bc.RESULTS / corpus.res("cheap_baseline.json")).read_text())["cheap_baseline"]["C"]
    sc = StandardScaler().fit(x[tr])
    clf = LogisticRegression(C=c, max_iter=3000).fit(sc.transform(x[tr]), y[tr])
    assert list(clf.classes_) == keep
    col = {k: j for j, k in enumerate(keep)}
    ours = conformal(clf.predict_proba(sc.transform(x[cal])), np.array([col[v] for v in y[cal]]),
                     clf.predict_proba(sc.transform(x[te])), np.array([col[v] for v in y[te]]))
    mapie = json.loads((bc.RESULTS / corpus.res("conformal_cheap.json")).read_text())
    for s, m in (("lac", "lac"), ("aps_nonrandom", "aps")):
        print(f"{s}: ours coverage {ours[s]['coverage']:.4f} review {ours[s]['human_review_load']:.4f} | "
              f"MAPIE {m} coverage {mapie[m]['coverage']:.4f} review {mapie[m]['human_review_load']:.4f}")


def compare(names):
    res = {n: json.loads((bc.RESULTS / corpus.res(f"strong_{n}.json")).read_text())["summary"] for n in names}
    base = res[names[0]]
    verdict = {}
    for n in names[1:]:
        verdict[n] = {}
        for m in ("alpha_leaf", "macro_f1_leaf"):
            diff = res[n][m]["mean"] - base[m]["mean"]
            margin = max(res[n][m]["sd"], base[m]["sd"])
            verdict[n][m] = {"difference": diff, "margin": margin,
                             "call": "better" if diff > margin else "worse" if diff < -margin else "no difference"}
        print(n, {m: f"{v['difference']:+.4f} (margin {v['margin']:.4f}): {v['call']}" for m, v in verdict[n].items()})
    (bc.RESULTS / corpus.res("step6_comparison.json")).write_text(json.dumps({"baseline": names[0], "verdict": verdict,
                                                                  "summaries": res}, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--name")
    ap.add_argument("--posthoc", action="store_true")
    ap.add_argument("--temperature_only", action="store_true", help="posthoc with tau fixed at 0")
    ap.add_argument("runs", nargs="*")
    args = ap.parse_args()
    if args.check:
        return check()
    if args.compare:
        return compare(args.compare)

    units, keep, _, _, parent = bc.load()
    g = gold()
    y_tr = [r["label"] for r in g.values() if r["split"] == "train"]
    prevalence = np.array([np.mean([v == c for v in y_tr]) for c in keep])
    if args.posthoc or args.temperature_only:
        taus = [0.0] if args.temperature_only else None
        run_dirs = [posthoc(r, g, keep, prevalence, taus) for r in args.runs]
    else:
        run_dirs = args.runs
    runs = {r: score_run(r, g, keep, parent, prevalence) for r in run_dirs}
    summary = summarise_runs(runs)
    (bc.RESULTS / corpus.res(f"strong_{args.name}.json")).write_text(json.dumps({"runs": runs, "summary": summary}, indent=2),
                                                        encoding="utf-8")
    for m, v in summary.items():
        if not m.startswith("ppi_"):
            print(f"{m:26} {v['mean']:.4f} +/- {v['sd']:.4f}")


if __name__ == "__main__":
    main()
