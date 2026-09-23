"""Step 2: build the handbook 5 parent-child table from the published handbook.

Source is the "Category Scheme" section of the Manifesto Coding Instructions,
5th edition PDF. Parents come from the document structure, never from the
digits of a code:
  DOMAIN n: <name>                       -> domain node, no parent
  [xxx <name>, comprised of:]            -> category whose children follow
      xxx.y <name>  (indented)           -> subcategory, parent is the open category
  xxx <name>                             -> category, parent is the open domain
                                            (203 and 204 are printed indented after
                                            the 202 block; the overview page places
                                            them directly under Domain 2)
  000 No meaningful category applies     -> listed outside all domains, no parent

The definition is the text under each heading, verbatim apart from joining
wrapped lines. Categories that only exist as "comprised of" containers have
no definition of their own in the handbook; they carry only whatever note the
handbook prints under them, and are otherwise empty.

Requires pdftotext (poppler), which ships with Git for Windows.

Writes two files:
  data/codeframe_hb5.csv             code, parent_code, definition (not committed:
                                     the definitions are handbook text, and the
                                     Manifesto Project's terms forbid redistribution)
  data/codeframe_hb5_structure.csv   code, parent_code (committed; the pipeline
                                     reads parents from this file)
If the structure file already exists, the new parse must match it exactly.

Usage: python build_codeframe.py [--revision 2021|2026]
"""

import argparse
import csv
import difflib
import re
import subprocess
import unicodedata
import urllib.request

import mp_api

HANDBOOKS = {
    "2021": ("handbook_2021_version_5.pdf", "Procedure for Training"),
    "2026": ("handbook_2026_version_5.pdf", "Training Test"),
}
HANDBOOK_URL = "https://manifesto-project.wzb.eu/down/papers/"
HANDBOOK_DIR = mp_api.CACHE_DIR / "handbook"

DOMAIN_RE = re.compile(r"^DOMAIN (\d): (.+)$")
CONTAINER_RE = re.compile(r"^\[(\d{3}) (.+), comprised of:\]$")
SUB_RE = re.compile(r"^\s{2,}(\d{3}\.\d) (.+)$")
CAT_RE = re.compile(r"^\s*(\d{3}) ([A-Z].*)$")
FOOTER_RE = re.compile(r"Handbook Series|^\s*\d*\s*$|^\*+$")


def handbook_text(revision):
    pdf_name, end_marker = HANDBOOKS[revision]
    pdf = HANDBOOK_DIR / pdf_name
    if not pdf.exists():
        HANDBOOK_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HANDBOOK_URL + pdf_name, pdf)
    txt = pdf.with_suffix(".txt")
    subprocess.run(["pdftotext", "-enc", "UTF-8", "-layout", str(pdf), str(txt)],
                   check=True)
    lines = txt.read_text(encoding="utf-8").splitlines()
    start = lines.index("Category Scheme")
    end = next(i for i in range(start, len(lines))
               if lines[i].strip().startswith(end_marker))
    return lines[start + 1 : end]


def clean(text_lines, hyphenated=frozenset()):
    """Join wrapped lines. A word broken across lines ("democ- racy") is
    rejoined, unless the handbook also writes it hyphenated in running text."""
    text = " ".join(l.strip() for l in text_lines if l.strip())
    text = unicodedata.normalize("NFKC", text)

    def join(m):
        word = f"{m.group(1)}-{m.group(2)}"
        return word if word.lower() in hyphenated else m.group(1) + m.group(2)

    text = re.sub(r"(\w+)- (\w+)", join, text)
    return re.sub(r"\s+", " ", text).strip()


def inline_hyphenated(lines):
    words = set()
    for l in lines:
        words.update(w.lower() for w in re.findall(r"\w+-\w+", unicodedata.normalize("NFKC", l)))
    return frozenset(words)


def parse(lines):
    rows = []          # dicts: code, parent_code, title, body (list of lines)
    domain = None
    container = None
    current = None
    for line in lines:
        if FOOTER_RE.search(line):
            continue
        stripped = line.strip()
        if m := DOMAIN_RE.match(stripped):
            domain = f"domain_{m.group(1)}"
            container = None
            current = {"code": domain, "parent_code": "", "title": m.group(2), "body": []}
            rows.append(current)
        elif m := CONTAINER_RE.match(stripped):
            container = m.group(1)
            current = {"code": container, "parent_code": domain,
                       "title": m.group(2), "body": []}
            rows.append(current)
        elif m := SUB_RE.match(line):
            if container is None:
                raise ValueError(f"subcategory outside a container: {line!r}")
            current = {"code": m.group(1), "parent_code": container,
                       "title": m.group(2).strip(), "body": []}
            rows.append(current)
        elif m := CAT_RE.match(line):
            container = None
            # 000 is listed after the seven domains and belongs to none of them.
            parent = "" if m.group(1) == "000" else domain
            current = {"code": m.group(1), "parent_code": parent,
                       "title": m.group(2).strip(), "body": []}
            rows.append(current)
        elif current is not None:
            current["body"].append(line)
    hyphenated = inline_hyphenated(lines)
    for r in rows:
        r["definition"] = clean(r.pop("body"), hyphenated)
    return rows


def cross_check(rows):
    """Compare against the API codebook. Reports, never edits."""
    book = mp_api.call("get_core_codebook", {"key": "MPDS2026a"})
    header = book[0]
    api = {r[header.index("code")]: dict(zip(header, r)) for r in book[1:]
           if r[0] in ("main", "hb5")}
    ours = {r["code"]: r for r in rows}
    domain_of = {}
    for r in rows:
        p = r
        while p["parent_code"]:
            p = ours[p["parent_code"]]
        domain_of[r["code"]] = p["code"] if p["code"].startswith("domain_") else ""

    problems = []
    for code in sorted(set(api) - set(ours)):
        problems.append(f"in API codebook, missing from handbook: {code}")
    for code, r in ours.items():
        if code.startswith("domain_"):
            continue
        if code not in api:
            problems.append(f"in handbook, missing from API codebook: {code}")
            continue
        want = api[code]["domain_code"]
        got = domain_of[code].removeprefix("domain_") if domain_of[code] else "0"
        if want != got:
            problems.append(f"domain mismatch for {code}: handbook {got}, API {want}")

    # Not a source of parents, only a consistency check on the parse.
    for r in rows:
        if "." in r["code"] and r["code"].split(".")[0] != r["parent_code"]:
            problems.append(f"parse check: {r['code']} placed under {r['parent_code']}")

    drift = []
    for code, r in ours.items():
        if code in api and r["definition"]:
            a = clean([api[code]["description_md"].replace("-   ", "• ")])
            ratio = difflib.SequenceMatcher(None, r["definition"], a, autojunk=False).ratio()
            if ratio < 0.97:
                drift.append((round(ratio, 3), code))
    return problems, sorted(drift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", choices=sorted(HANDBOOKS), default="2021")
    args = ap.parse_args()

    rows = parse(handbook_text(args.revision))
    out = mp_api.ROOT / "data" / "codeframe_hb5.csv"
    if args.revision != "2021":
        out = out.with_name(f"codeframe_hb5_{args.revision}.csv")
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "parent_code", "definition"])
        for r in rows:
            w.writerow([r["code"], r["parent_code"], r["definition"]])

    structure = out.with_name(out.stem + "_structure.csv")
    lines = ["code,parent_code"] + [f"{r['code']},{r['parent_code']}" for r in rows]
    new = "\n".join(lines) + "\n"
    if structure.exists() and structure.read_text(encoding="utf-8") != new:
        raise SystemExit(f"parsed tree differs from committed {structure.name}; not overwriting")
    structure.write_text(new, encoding="utf-8")

    kinds = {
        "domains": sum(r["code"].startswith("domain_") for r in rows),
        "categories": sum(re.fullmatch(r"\d{3}", r["code"]) is not None for r in rows),
        "subcategories": sum("." in r["code"] for r in rows),
        "empty definitions": sum(not r["definition"] for r in rows),
    }
    print(f"wrote {len(rows)} rows to {out}: {kinds}")
    problems, drift = cross_check(rows)
    print("cross-check problems:", problems or "none")
    print("definitions differing from the API codebook text (ratio < 0.97):")
    for ratio, code in drift:
        print(f"  {code}: {ratio}")


if __name__ == "__main__":
    main()
