"""Step 1b: count what exists per category, per language and handbook version.

Reads data/raw/quasi_sentences.jsonl (from fetch_slice.py) and
data/codeframe_hb5_structure.csv (from build_codeframe.py). Identical documents are
counted once.

Unit types in cmp_code:
  NA    text the coders did not code (titles, tables of contents, etc.)
  H     headings, marked as such under handbooks 4 and 5
  000   coded, but no category applies
  other a code from the handbook 5 codeframe, or an off-scheme code

Writes data/label_counts.csv and prints the tables used in FINDINGS.md.
"""

import collections
import csv

import corpus
import mp_api

DATA = mp_api.ROOT / "data"
SMALL = 100


def load_codeframe():
    parent = corpus.codeframe()
    has_child = set(parent.values())
    codes = [c for c in parent if not c.startswith("domain_")]
    leaves = [c for c in codes if c not in has_child]
    containers = [c for c in codes if c in has_child]
    return parent, leaves, containers


def load_units():
    """All units, with identical documents kept once (see corpus.dedupe)."""
    docs, dropped = corpus.dedupe(corpus.load_documents())
    print(f"dropped {len(dropped)} duplicate documents:",
          ", ".join(f"{k} (= {v})" for k, v in sorted(dropped.items())))
    for key in sorted(docs):
        yield from docs[key]


def main():
    parent, leaves, containers = load_codeframe()
    known = set(leaves) | set(containers)

    by_slice = collections.defaultdict(collections.Counter)   # (lang, hb) -> code counts
    docs = collections.defaultdict(set)
    countries = collections.defaultdict(set)
    for r in load_units():
        key = (r["language"], r["handbook"])
        by_slice[key][r["cmp_code"]] += 1
        docs[key].add(r["manifesto_id"])
        countries[key].add(r["country"])

    print("== Units per language and handbook version ==")
    print("lang     hb   docs  countries   total      NA       H    000   in-scheme  off-scheme")
    for key in sorted(by_slice, key=lambda k: (k[0], str(k[1]))):
        c = by_slice[key]
        total = sum(c.values())
        scheme = sum(n for code, n in c.items() if code in known and code != "000")
        off = sum(n for code, n in c.items() if code not in known and code not in ("NA", "H"))
        print(f"{key[0]:8} {str(key[1]):4} {len(docs[key]):5} {len(countries[key]):10} "
              f"{total:7} {c['NA']:7} {c['H']:7} {c['000']:6} {scheme:11} {off:11}")

    out_rows = []
    for lang in ("english", "dutch"):
        c = by_slice.get((lang, "5"), collections.Counter())
        print(f"\n== {lang}, handbook 5 ==")
        print("countries:", sorted(countries[(lang, "5")]))
        off = {k: v for k, v in c.items() if k not in known and k not in ("NA", "H")}
        print("off-scheme codes:", dict(sorted(off.items(), key=lambda kv: -kv[1])) or "none")

        leaf_counts = sorted(((c.get(code, 0), code) for code in leaves if code != "000"),
                             reverse=True)
        print(f"leaf counts, sorted ({len(leaf_counts)} leaves, 000 excluded):")
        for n, code in leaf_counts:
            print(f"  {code:6} {n}")
            out_rows.append([lang, "5", "leaf", code, n])
        small = [code for n, code in leaf_counts if n < SMALL]
        zero = [code for n, code in leaf_counts if n == 0]
        print(f"leaves under {SMALL}: {len(small)} of {len(leaf_counts)}; zero: {len(zero)} {zero}")

        print("subcategory use (bare container code vs its subcategories):")
        for cont in containers:
            kids = [code for code in leaves if parent[code] == cont]
            kid_n = {k: c.get(k, 0) for k in kids}
            print(f"  {cont}: bare={c.get(cont, 0)}  " +
                  "  ".join(f"{k}={n}" for k, n in kid_n.items()))

        cat = collections.Counter()
        for code, n in c.items():
            if code in known and code != "000":
                cat[corpus.category_of(code, parent)] += n
        cats = [code for code in known if code not in ("000",) and corpus.category_of(code, parent) == code]
        cat_small = [code for code in cats if cat.get(code, 0) < SMALL]
        print(f"category level (subcategories collapsed): {len(cat_small)} of {len(cats)} under {SMALL}")
        for code in cats:
            out_rows.append([lang, "5", "category", code, cat.get(code, 0)])

    print("\n== Widening options, English, category level (subcategories collapsed) ==")
    options = {
        "hb5 only": ["5"],
        "hb4 + hb5": ["4", "5"],
        "all handbooks": sorted({hb for (lang, hb) in by_slice if lang == "english"}),
    }
    cats = sorted(code for code in known if code != "000" and corpus.category_of(code, parent) == code)
    for name, hbs in options.items():
        cat = collections.Counter()
        n_docs = 0
        for hb in hbs:
            n_docs += len(docs.get(("english", hb), ()))
            for code, n in by_slice.get(("english", hb), {}).items():
                if code in known and code != "000":
                    cat[corpus.category_of(code, parent)] += n
        total = sum(cat.values())
        under = sum(cat.get(code, 0) < SMALL for code in cats)
        med = sorted(cat.get(code, 0) for code in cats)[len(cats) // 2]
        print(f"  {name:14} docs={n_docs:4} units={total:7} categories<{SMALL}: {under:2} of {len(cats)}  median={med}")

    with (DATA / "label_counts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["language", "handbook", "level", "code", "n"])
        w.writerows(out_rows)


if __name__ == "__main__":
    main()
