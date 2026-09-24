"""Write the modelling set as one self-contained file, so a GPU machine needs
only this file and train_encoder.py (no API key, no raw download).

Output: data/raw/modelling_en_hb5.jsonl, one line per modelled unit:
  manifesto_id, pos, country, split, label, text
It holds Manifesto Project text, so it stays out of git (data/raw/ is ignored)
and must not be shared further.
"""

import csv
import json

import corpus

OUT = corpus.MODELLING


def main():
    units, _ = corpus.coded_units()
    keep, _, _ = corpus.eligible_leaves(units, corpus.codeframe())
    keep = set(keep)
    split_of = {r["manifesto_id"]: r["split"]
                for r in csv.DictReader(corpus.SPLITS.open(encoding="utf-8"))}
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for u in units:
            if u["cmp_code"] in keep:
                f.write(json.dumps({"manifesto_id": u["manifesto_id"], "pos": u["pos"],
                                    "country": u["country"], "split": split_of[u["manifesto_id"]],
                                    "label": u["cmp_code"], "text": u["text"] or ""},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} units to {OUT}")


if __name__ == "__main__":
    main()
