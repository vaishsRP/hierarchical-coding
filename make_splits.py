"""Fix the train / dev / calib / test split once, before any model is trained.

Unit of assignment is the party: all of a party's manifestos land in the same
split, because parties reuse text across elections and a document-level split
would let a model memorise a party's boilerplate. Within each country,
parties are assigned largest first to whichever split is furthest below its
target share of coded units. Ties are broken by a seeded shuffle.

  train  60%  model fitting
  dev    10%  hyperparameter choice and early stopping
  calib  15%  conformal calibration (step 7), untouched until then
  test   15%  reported numbers

Writes data/splits.csv (manifesto_id, party, country, split, units). The file
is not committed (it lists Manifesto Project ids); its SHA-256 is recorded in
SPEC.md so a rerun can be checked against it.
"""

import collections
import csv
import hashlib
import random

import corpus

SEED = 20260923
TARGETS = {"train": 0.60, "dev": 0.10, "calib": 0.15, "test": 0.15}


def assign(units):
    size = collections.Counter()
    country = {}
    docs = collections.defaultdict(set)
    for u in units:
        party = u["manifesto_id"].split("_")[0]
        size[party] += 1
        country[party] = u["country"]
        docs[party].add(u["manifesto_id"])

    rng = random.Random(SEED)
    split_of = {}
    for c in sorted(set(country.values())):
        parties = [p for p in size if country[p] == c]
        rng.shuffle(parties)
        parties.sort(key=lambda p: -size[p])        # stable: shuffle breaks ties
        total = sum(size[p] for p in parties)
        got = collections.Counter()
        for p in parties:
            order = list(TARGETS)
            rng.shuffle(order)
            best = max(order, key=lambda s: TARGETS[s] * total - got[s])
            split_of[p] = best
            got[best] += size[p]
    return split_of, size, country, docs


def main():
    units, _ = corpus.coded_units("english", "5")
    split_of, size, country, docs = assign(units)

    doc_units = collections.Counter(u["manifesto_id"] for u in units)
    rows = sorted((m, p, country[p], split_of[p], doc_units[m])
                  for p in split_of for m in docs[p])
    with (corpus.DATA / "splits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["manifesto_id", "party", "country", "split", "units"])
        w.writerows(rows)

    digest = hashlib.sha256((corpus.DATA / "splits.csv").read_bytes()).hexdigest()
    print(f"splits.csv sha256 {digest}")

    total = sum(size.values())
    print("split  parties  docs   units  share")
    for s in TARGETS:
        ps = [p for p in split_of if split_of[p] == s]
        n = sum(size[p] for p in ps)
        print(f"{s:6} {len(ps):7} {sum(len(docs[p]) for p in ps):5} {n:7}  {n / total:.3f}")

    print("\ncountry share of each split:")
    by = collections.defaultdict(collections.Counter)
    for p, s in split_of.items():
        by[s][country[p]] += size[p]
    for s in TARGETS:
        n = sum(by[s].values())
        print(f"  {s:6}", ", ".join(f"{c} {v / n:.2f}" for c, v in by[s].most_common()))

    parent = corpus.codeframe()
    keep, _, _ = corpus.eligible_leaves(units, parent)
    per = collections.defaultdict(collections.Counter)
    for u in units:
        per[split_of[u["manifesto_id"].split("_")[0]]][u["cmp_code"]] += 1
    print("\nper split, eligible leaves with support: min, and count under 10")
    for s in TARGETS:
        sup = [per[s][c] for c in keep]
        low = sorted((per[s][c], c) for c in keep)[:5]
        print(f"  {s:6} min={min(sup)}  under10={sum(x < 10 for x in sup)}  lowest={low}")


if __name__ == "__main__":
    main()
