"""Shared view of the corpus. Every later step loads units through here so the
label counts, the splits and the models all see the same data.

Rules fixed in SPEC.md ("Decisions fixed after steps 1 and 2"):
  - identical documents are kept once (first manifesto_id in sort order)
  - NA and H units are not coded and are dropped
  - a leaf is modelled if it has at least MIN_LEAF_UNITS units in the
    deduplicated slice; units of other leaves, including 000, are set aside
  - parents follow the handbook table data/codeframe_hb5_structure.csv, so 202.2,
    605.2 and 703.2 sit under 202, 605 and 703 (not under 000 as in the
    Manifesto Project dataset)
"""

import collections
import csv
import hashlib
import json
import os

import mp_api

DATA = mp_api.ROOT / "data"
SENTENCES = mp_api.CACHE_DIR / "quasi_sentences.jsonl"
NOT_CODED = {"NA", "H"}
MIN_LEAF_UNITS = 100

# Language of the run: set HC_LANG=dutch for the Dutch pipeline. English keeps
# the original file names; other languages get a suffix on every output.
LANG = os.environ.get("HC_LANG", "english")
TAG = "" if LANG == "english" else f"_{LANG}"
SPLITS = DATA / f"splits{TAG}.csv"
MODELLING = mp_api.CACHE_DIR / f"modelling_{'en' if LANG == 'english' else 'nl'}_hb5.jsonl"


def res(name):
    """Result file name for the current language: cheap_baseline.json -> cheap_baseline_dutch.json."""
    stem, dot, ext = name.rpartition(".")
    return f"{stem}{TAG}.{ext}"


def codeframe():
    rows = list(csv.DictReader((DATA / "codeframe_hb5_structure.csv").open(encoding="utf-8")))
    return {r["code"]: r["parent_code"] for r in rows}


def ancestors(code, parent):
    """[category, domain] for a leaf; 000 has none."""
    out = []
    p = parent.get(code, "")
    while p:
        out.append(p)
        p = parent.get(p, "")
    return out


def category_of(code, parent):
    p = parent.get(code, "")
    return p if p and not p.startswith("domain_") else code


def domain_of(code, parent):
    chain = [code] + ancestors(code, parent)
    return next((c for c in chain if c.startswith("domain_")), "none")


def load_documents(language=None, handbook=None):
    """{manifesto_id: [unit, ...]} with every unit in document order."""
    docs = collections.defaultdict(list)
    with SENTENCES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if language and r["language"] != language:
                continue
            if handbook and r["handbook"] != handbook:
                continue
            r["cmp_code"] = str(r["cmp_code"] or "NA").strip()
            docs[r["manifesto_id"]].append(r)
    return dict(docs)


def dedupe(docs):
    """Keep one document per identical (text, code) sequence."""
    seen = {}
    dropped = {}
    for key in sorted(docs):
        sig = hashlib.sha1(json.dumps(
            [(u["text"], u["cmp_code"]) for u in docs[key]]).encode("utf-8")).hexdigest()
        if sig in seen:
            dropped[key] = seen[sig]
        else:
            seen[sig] = key
    return {k: v for k, v in docs.items() if k not in dropped}, dropped


def coded_units(language=None, handbook="5"):
    """Deduplicated coded units (NA and H removed) plus the dropped duplicates."""
    docs, dropped = dedupe(load_documents(language or LANG, handbook))
    units = [u for key in sorted(docs) for u in docs[key] if u["cmp_code"] not in NOT_CODED]
    return units, dropped


def eligible_leaves(units, parent):
    counts = collections.Counter(u["cmp_code"] for u in units)
    leaves = {c for c in parent if not c.startswith("domain_")} - set(parent.values())
    keep = sorted(c for c in leaves if c != "000" and counts[c] >= MIN_LEAF_UNITS)
    sparse = sorted((c for c in leaves if c not in keep), key=lambda c: -counts[c])
    return keep, sparse, counts
