"""Step 3: cheap flat baseline. Sentence embeddings plus multinomial logistic
regression over the 63 modelled leaves, no hierarchy information.

Protocol (fixed in SPEC.md before this was run):
  - fit on train, choose C on dev log-loss, report on test
  - no class reweighting
  - headline: Krippendorff's alpha (nominal, human vs model) and macro F1
    over the modelled leaves; category and domain scores are derived from
    the predicted leaf through data/codeframe_hb5_structure.csv

Two reference rows are reported alongside, not as baselines to beat:
majority class (the alpha floor) and TF-IDF plus the same classifier.

Writes results/cheap_baseline.json, results/cheap_baseline_per_leaf.csv, and
test-set probabilities to data/raw/preds_cheap_test.npz for step 4.
"""

import collections
import csv
import hashlib
import json
import platform
import time

import krippendorff
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

import corpus

# English: bge small. Other languages: multilingual e5 small, same size of
# encoder (12 layers, 384 wide); e5 expects a "query: " prefix on every input.
EMBED_MODEL = "BAAI/bge-small-en-v1.5" if corpus.LANG == "english" else "intfloat/multilingual-e5-small"
EMBED_PREFIX = "" if corpus.LANG == "english" else "query: "
C_GRID = [0.01, 0.1, 1.0, 10.0]
RESULTS = corpus.mp_api.ROOT / "results"
EMB_DIR = corpus.mp_api.CACHE_DIR / "embeddings"
CHUNK = 5000


def load():
    units, _ = corpus.coded_units()
    parent = corpus.codeframe()
    keep, sparse, counts = corpus.eligible_leaves(units, parent)
    split_of = {r["manifesto_id"]: r["split"]
                for r in csv.DictReader(corpus.SPLITS.open(encoding="utf-8"))}
    modelled = [u for u in units if u["cmp_code"] in set(keep)]
    for u in modelled:
        u["split"] = split_of[u["manifesto_id"]]
    return modelled, keep, sparse, counts, parent


def embed(texts):
    """Embed in chunks cached on disk, so an interrupted run resumes."""
    from sentence_transformers import SentenceTransformer
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    tag = EMBED_MODEL.replace("/", "__")
    model = None
    parts = []
    texts = [EMBED_PREFIX + t for t in texts]
    for i in range(0, len(texts), CHUNK):
        chunk = texts[i : i + CHUNK]
        digest = hashlib.sha1("\n".join(chunk).encode("utf-8")).hexdigest()[:12]
        path = EMB_DIR / f"{tag}_{i:07d}_{digest}.npy"
        if not path.exists():
            if model is None:
                model = SentenceTransformer(EMBED_MODEL, device="cpu")
            t = time.time()
            np.save(path, model.encode(chunk, batch_size=64, normalize_embeddings=True))
            print(f"  embedded {i + len(chunk)}/{len(texts)} ({time.time() - t:.0f}s)", flush=True)
        parts.append(np.load(path))
    return np.vstack(parts)


def alpha_nominal(a, b):
    """Krippendorff's alpha, nominal, two coders, no missing values.
    Written out to cross-check the krippendorff package."""
    values = sorted(set(a) | set(b))
    idx = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for x, y in zip(a, b):
        o[idx[x], idx[y]] += 1
        o[idx[y], idx[x]] += 1
    n_c = o.sum(axis=1)
    n = n_c.sum()
    d_o = (n - np.trace(o)) / n
    d_e = (n * n - (n_c ** 2).sum()) / (n * (n - 1))
    return 1 - d_o / d_e


def alpha(a, b):
    codes = {c: i for i, c in enumerate(sorted(set(a) | set(b)))}
    data = np.array([[codes[x] for x in a], [codes[x] for x in b]], dtype=float)
    pkg = krippendorff.alpha(reliability_data=data, level_of_measurement="nominal")
    own = alpha_nominal(a, b)
    assert abs(pkg - own) < 1e-6, (pkg, own)
    return float(pkg)


def scores(y_true, y_pred, keep, parent):
    cat_t = [corpus.category_of(c, parent) for c in y_true]
    cat_p = [corpus.category_of(c, parent) for c in y_pred]
    dom_t = [corpus.domain_of(c, parent) for c in y_true]
    dom_p = [corpus.domain_of(c, parent) for c in y_pred]
    return {
        "alpha_leaf": alpha(y_true, y_pred),
        "macro_f1_leaf": float(f1_score(y_true, y_pred, labels=keep, average="macro", zero_division=0)),
        "accuracy_leaf": float(accuracy_score(y_true, y_pred)),
        "alpha_category": alpha(cat_t, cat_p),
        "accuracy_category": float(accuracy_score(cat_t, cat_p)),
        "alpha_domain": alpha(dom_t, dom_p),
        "accuracy_domain": float(accuracy_score(dom_t, dom_p)),
    }


def fit_select(x_tr, y_tr, x_dev, y_dev, labels):
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000)
        clf.fit(x_tr, y_tr)
        loss = log_loss(y_dev, clf.predict_proba(x_dev), labels=clf.classes_)
        print(f"    C={c:<6} dev log-loss {loss:.4f}", flush=True)
        if best is None or loss < best[0]:
            best = (loss, c, clf)
    return best


def main():
    units, keep, sparse, counts, parent = load()
    by = collections.defaultdict(list)
    for i, u in enumerate(units):
        by[u["split"]].append(i)
    y = np.array([u["cmp_code"] for u in units])
    texts = [u["text"] or "" for u in units]
    print(f"{len(units)} modelled units, {len(keep)} leaves; "
          + ", ".join(f"{s} {len(by[s])}" for s in ("train", "dev", "calib", "test")))

    tr, dev, te = by["train"], by["dev"], by["test"]
    out = {"protocol": {"embed_model": EMBED_MODEL, "c_grid": C_GRID,
                        "selection": "dev log-loss", "class_weight": None,
                        "min_leaf_units": corpus.MIN_LEAF_UNITS},
           "data": {"modelled_units": len(units), "leaves": len(keep),
                    "split_units": {s: len(v) for s, v in by.items()},
                    "sparse_set_aside": {c: counts[c] for c in sparse}},
           "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__}}

    majority = collections.Counter(y[tr]).most_common(1)[0][0]
    out["majority_class"] = scores(list(y[te]), [majority] * len(te), keep, parent)
    print("majority:", out["majority_class"], flush=True)

    print("TF-IDF reference:", flush=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    xt = vec.fit_transform([texts[i] for i in tr])
    loss, c, clf = fit_select(xt, y[tr], vec.transform([texts[i] for i in dev]), y[dev], keep)
    pred = clf.predict(vec.transform([texts[i] for i in te]))
    out["tfidf_reference"] = {"C": c, "dev_log_loss": loss, **scores(list(y[te]), list(pred), keep, parent)}
    print("tfidf:", out["tfidf_reference"], flush=True)

    print(f"embedding with {EMBED_MODEL}:", flush=True)
    x = embed(texts)
    scaler = StandardScaler().fit(x[tr])
    xs = scaler.transform(x)
    loss, c, clf = fit_select(xs[tr], y[tr], xs[dev], y[dev], keep)
    proba = clf.predict_proba(xs[te])
    pred = clf.classes_[proba.argmax(axis=1)]
    out["cheap_baseline"] = {"C": c, "dev_log_loss": loss,
                             "test_log_loss": float(log_loss(y[te], proba, labels=clf.classes_)),
                             **scores(list(y[te]), list(pred), keep, parent)}
    print("cheap baseline:", out["cheap_baseline"], flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / corpus.res("cheap_baseline.json")).write_text(json.dumps(out, indent=2), encoding="utf-8")

    f1 = f1_score(y[te], pred, labels=keep, average=None, zero_division=0)
    support = collections.Counter(y[te])
    with (RESULTS / corpus.res("cheap_baseline_per_leaf.csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "category", "domain", "train_units", "test_units", "f1"])
        train_n = collections.Counter(y[tr])
        for code, score in sorted(zip(keep, f1), key=lambda t: -t[1]):
            w.writerow([code, corpus.category_of(code, parent), corpus.domain_of(code, parent),
                        train_n[code], support[code], round(float(score), 4)])

    np.savez_compressed(corpus.mp_api.CACHE_DIR / corpus.res("preds_cheap_test.npz"),
                        manifesto_id=np.array([units[i]["manifesto_id"] for i in te]),
                        pos=np.array([units[i]["pos"] for i in te]),
                        y_true=y[te], classes=clf.classes_, proba=proba)
    print("wrote results/" + corpus.res("cheap_baseline.json"))


if __name__ == "__main__":
    main()
