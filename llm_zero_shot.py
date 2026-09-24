"""LLM comparison: zero shot coding of English test sentences with an open
weights model on Groq's free tier, scored against the other models on the
same sentences. Protocol in SPEC.md ("LLM comparison" and its dated note).

Two samples: a random 2,000 test sentences (seed 0), and every test sentence
whose code is one of four "against" stances. The prompt lists the 63 codes
with their handbook names only: no definitions and no training sentences, so
only test sentences leave this machine.

Every batch response is cached under data/raw/llm/, so the run resumes after
a rate limit or an interruption. The API key is read from groq_apikey.txt
(gitignored).

Writes results/llm_zero_shot.json.
"""

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import baseline_cheap as bc
import corpus

MODEL = "openai/gpt-oss-120b"
URL = "https://api.groq.com/openai/v1/chat/completions"
KEY = corpus.mp_api.ROOT / "groq_apikey.txt"
CACHE = corpus.mp_api.CACHE_DIR / "llm"
BATCH = 20
SAMPLE = 2000
STANCE = {"601.2": "602.2", "505": "504", "110": "108", "105": "104"}
STANCE_NAMES = {"601.2": "Against immigration", "505": "Against expanding welfare",
                "110": "Against the EU", "105": "Against the military"}


def prompt(names, keep):
    codes = "\n".join(f"{c}: {names[c]}" for c in keep)
    return ("You code sentences from party election manifestos with the Manifesto Project scheme. "
            "Give each sentence exactly one code from this list:\n" + codes +
            "\n\nAnswer with JSON only, mapping each sentence number to its code, "
            'for example {"1": "504", "2": "411"}.')


def call(system, sentences, key):
    user = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    body = json.dumps({"model": MODEL, "temperature": 0, "reasoning_effort": "low",
                       "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "hc-research/1.0"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                wait = float(e.headers.get("retry-after") or 20 * (attempt + 1))
                print(f"  {e.code}, waiting {wait:.0f}s", flush=True)
                time.sleep(wait + 1)
                continue
            raise
    raise RuntimeError("too many retries")


def code_batch(system, batch, key):
    """Returns codes for one batch, cached on disk by content."""
    CACHE.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(json.dumps([MODEL, system, batch]).encode()).hexdigest()[:16]
    path = CACHE / f"{h}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raw = call(system, batch, key)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        parsed = json.loads(m.group(0)) if m else {}
    codes = [str(parsed.get(str(i + 1), "")).strip() for i in range(len(batch))]
    path.write_text(json.dumps(codes), encoding="utf-8")
    return codes


def other_models(ids):
    """Predictions of the cheap, flat and definitions models for these unit ids."""
    out = {}
    z = np.load(corpus.mp_api.CACHE_DIR / "preds_cheap_test.npz", allow_pickle=True)
    cheap_ids = [f"{m}#{p}" for m, p in zip(z["manifesto_id"], z["pos"])]
    pos = {u: i for i, u in enumerate(cheap_ids)}
    out["cheap"] = [[str(z["classes"][z["proba"][pos[u]].argmax()]) for u in ids]]
    for name, pattern in (("deberta_flat", "runs/kaggle_full/runs/deberta_s{}/probs.npz"),
                          ("deberta_definitions", "runs/kaggle_hc-structured-defs/runs/defs_s{}/probs.npz")):
        runs = []
        for s in range(3):
            z = np.load(corpus.mp_api.ROOT / pattern.format(s))
            pos = {u: i for i, u in enumerate(z["test_ids"])}
            labels = list(z["labels"])
            runs.append([labels[z["test_proba"][pos[u]].argmax()] for u in ids])
        out[name] = runs
    return out


def pooled_bias(y, pred, keep):
    """Share of the sample's content put in the wrong category, pooled over the
    whole sample (too few sentences per manifesto for per document shares)."""
    true = np.array([np.mean(y == c) for c in keep])
    got = np.array([np.mean(np.asarray(pred) == c) for c in keep])
    return float(np.abs(got - true).sum() / 2)


def score(y, preds, years, keep, parent):
    res = [bc.scores(list(y), list(p), keep, parent) for p in preds]
    out = {k: float(np.mean([r[k] for r in res])) for k in ("alpha_leaf", "macro_f1_leaf", "accuracy_leaf", "alpha_domain")}
    out["pooled_wrong_category"] = float(np.mean([pooled_bias(y, p, keep) for p in preds]))
    new = np.array([yr == "2024" for yr in years])
    for tag, m in (("before_2024", ~new), ("2024", new)):
        out[f"accuracy_{tag}"] = float(np.mean([np.mean(np.array(p)[m] == y[m]) for p in preds]))
    out["accuracy_2024_n"] = int(new.sum())
    return out


def main():
    key = KEY.read_text(encoding="utf-8").strip()
    units, keep, _, _, parent = bc.load()
    names = json.loads((corpus.mp_api.ROOT / "app" / "assets" / "category_names.json").read_text(encoding="utf-8"))
    system = prompt(names, keep)
    test = [u for u in units if u["split"] == "test"]
    rng = np.random.default_rng(0)
    random_idx = sorted(rng.choice(len(test), SAMPLE, replace=False))
    samples = {"random": [test[i] for i in random_idx],
               "stance": [u for u in test if u["cmp_code"] in STANCE]}

    results = {"model": MODEL, "samples": {}}
    for name, rows in samples.items():
        texts = [r["text"] or "" for r in rows]
        llm = []
        for b in range(0, len(texts), BATCH):
            llm += code_batch(system, texts[b : b + BATCH], key)
            print(f"{name}: {min(b + BATCH, len(texts))}/{len(texts)}", flush=True)
        ids = [f"{r['manifesto_id']}#{r['pos']}" for r in rows]
        y = np.array([r["cmp_code"] for r in rows])
        years = [r["manifesto_id"][-6:-2] for r in rows]
        preds = {"llm": [[c if c in keep else "invalid" for c in llm]], **other_models(ids)}
        block = {"sentences": len(rows), "llm_invalid_answers": int(sum(c not in keep for c in llm))}
        for m, p in preds.items():
            block[m] = score(y, p, years, keep, parent)
            if name == "stance":
                block[m]["stance"] = {STANCE_NAMES[a]: {
                    "correct": float(np.mean([np.mean(np.array(r)[y == a] == a) for r in p])),
                    "flipped": float(np.mean([np.mean(np.array(r)[y == a] == b) for r in p]))}
                    for a, b in STANCE.items()}
        results["samples"][name] = block

    bc.RESULTS.mkdir(exist_ok=True)
    (bc.RESULTS / "llm_zero_shot.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    r = results["samples"]["random"]
    for m in ("llm", "cheap", "deberta_flat", "deberta_definitions"):
        print(f"{m:20} alpha {r[m]['alpha_leaf']:.3f}  macro F1 {r[m]['macro_f1_leaf']:.3f}  "
              f"acc {r[m]['accuracy_leaf']:.3f}  acc before 2024 {r[m]['accuracy_before_2024']:.3f}  "
              f"acc 2024 {r[m]['accuracy_2024']:.3f} (n={r[m]['accuracy_2024_n']})  "
              f"wrong category, pooled {r[m]['pooled_wrong_category']:.3f}")
    for m in ("llm", "cheap", "deberta_flat", "deberta_definitions"):
        print(m, {k: f"{v['correct']:.0%} right, {v['flipped']:.0%} flipped"
                  for k, v in results["samples"]["stance"][m]["stance"].items()})


if __name__ == "__main__":
    main()
