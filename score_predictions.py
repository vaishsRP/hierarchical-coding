"""Score one or more probability files with the step 3 metrics, so every
model is scored by the same code. Several files (one per seed) are reported
separately and as mean and standard deviation.

Usage: python score_predictions.py --name deberta runs/deberta_s0 runs/deberta_s1 ...
Writes results/<name>.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss

import baseline_cheap as bc
import corpus


def truth():
    rows = [json.loads(l) for l in (corpus.mp_api.CACHE_DIR / "modelling_en_hb5.jsonl").open(encoding="utf-8")]
    return {f"{r['manifesto_id']}#{r['pos']}": r["label"] for r in rows}


def score_run(run_dir, gold, keep, parent):
    z = np.load(Path(run_dir) / "probs.npz")
    labels = list(z["labels"])
    ids = z["test_ids"]
    proba = z["test_proba"]
    y = [gold[i] for i in ids]
    pred = [labels[j] for j in proba.argmax(axis=1)]
    res = bc.scores(y, pred, keep, parent)
    res["test_log_loss"] = float(log_loss(y, proba, labels=labels))
    log = json.loads((Path(run_dir) / "log.json").read_text())
    res["best_epoch"] = log.get("best_epoch")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()

    units, keep, _, _, parent = bc.load()
    gold = truth()
    runs = {r: score_run(r, gold, keep, parent) for r in args.runs}
    metrics = [k for k, v in next(iter(runs.values())).items() if isinstance(v, float)]
    summary = {m: {"mean": float(np.mean([r[m] for r in runs.values()])),
                   "sd": float(np.std([r[m] for r in runs.values()], ddof=1)) if len(runs) > 1 else None}
               for m in metrics}
    out = {"runs": runs, "summary": summary}
    bc.RESULTS.mkdir(exist_ok=True)
    (bc.RESULTS / f"{args.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for m, v in summary.items():
        sd = f" +/- {v['sd']:.4f}" if v["sd"] is not None else ""
        print(f"{m:20} {v['mean']:.4f}{sd}")


if __name__ == "__main__":
    main()
