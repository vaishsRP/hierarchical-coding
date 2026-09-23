"""Step 1a: download the English and Dutch coded quasi-sentences.

Pulls every annotated manifesto in these languages, under every handbook
version, so the counts can compare version 5 against the cost of widening.

Outputs
  data/manifesto_metadata.csv          one row per manifesto (committed)
  data/raw/quasi_sentences.jsonl       one row per quasi-sentence (not committed)
"""

import csv
import json

import mp_api

CORE_VERSION = "MPDS2026a"
CORPUS_VERSION = "2026-1"
LANGUAGES = {"english", "dutch"}
META_CHUNK = 100
TEXT_CHUNK = 5

DATA_DIR = mp_api.ROOT / "data"
META_CSV = DATA_DIR / "manifesto_metadata.csv"
SENTENCES = mp_api.CACHE_DIR / "quasi_sentences.jsonl"


def core_manifestos():
    rows = mp_api.call("get_core", {"key": CORE_VERSION})
    header, body = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    out = {}
    for r in body:
        key = f"{r[idx['party']]}_{r[idx['date']]}"
        out[key] = {
            "country": r[idx["country"]],
            "countryname": r[idx["countryname"]],
            "partyname": r[idx["partyname"]],
        }
    return out


def fetch_metadata(keys):
    items = []
    for batch in mp_api.chunks(keys, META_CHUNK):
        resp = mp_api.call("metadata", {"keys": batch, "version": CORPUS_VERSION})
        items.extend(resp["items"])
    return items


def main():
    core = core_manifestos()
    keys = sorted(core)
    print(f"{CORE_VERSION}: {len(keys)} manifestos in the core dataset")

    meta = fetch_metadata(keys)
    print(f"corpus {CORPUS_VERSION}: metadata for {len(meta)} manifestos")

    DATA_DIR.mkdir(exist_ok=True)
    fields = ["manifesto_id", "countryname", "partyname", "election_date",
              "language", "handbook", "annotations", "is_primary_doc",
              "translation_en", "title"]
    with META_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in meta:
            c = core.get(m["manifesto_id"], {})
            row = {k: m.get(k) for k in fields}
            row["countryname"] = c.get("countryname")
            row["partyname"] = c.get("partyname")
            w.writerow(row)

    wanted = [m for m in meta if m.get("annotations") and m.get("language") in LANGUAGES]
    by_key = {m["manifesto_id"]: m for m in wanted}
    print(f"annotated {sorted(LANGUAGES)} manifestos: {len(wanted)}")

    n_sent = 0
    with SENTENCES.open("w", encoding="utf-8") as out:
        for batch in mp_api.chunks(sorted(by_key), TEXT_CHUNK):
            resp = mp_api.call("texts_and_annotations",
                               {"keys": batch, "version": CORPUS_VERSION})
            for doc in resp["items"]:
                m = by_key[doc["key"]]
                country = core.get(doc["key"], {}).get("countryname")
                for pos, qs in enumerate(doc["items"]):
                    out.write(json.dumps({
                        "manifesto_id": doc["key"],
                        "pos": pos,
                        "country": country,
                        "language": m["language"],
                        "handbook": m.get("handbook"),
                        "cmp_code": qs.get("cmp_code"),
                        "text": qs.get("text"),
                    }, ensure_ascii=False) + "\n")
                    n_sent += 1
            print(f"  fetched {len(resp['items'])} docs, {n_sent} units so far")

    print(f"wrote {n_sent} units to {SENTENCES}")


if __name__ == "__main__":
    main()
