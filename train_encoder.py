"""Step 5: strong flat baseline. Fine-tune an encoder over all modelled
leaves at once, with no hierarchy information.

Self-contained: needs only data/raw/modelling_en_hb5.jsonl (from
export_modelling_set.py), torch and transformers, so it runs unchanged on a
GPU machine.

Protocol (fixed in SPEC.md before this was run):
  - model microsoft/deberta-v3-base, max 128 tokens
  - AdamW, lr 2e-5, batch 32, 4 epochs, linear schedule, 6% warmup,
    weight decay 0.01; no hyperparameter search
  - the epoch with the lowest dev log-loss is kept
  - seeds 0, 1, 2, reported separately and as mean and spread
  - no class reweighting, as in step 3

Writes <out>/probs.npz (dev, calib and test probabilities with unit ids)
and <out>/log.json. Scoring is done by score_predictions.py.

Usage: python train_encoder.py --seed 0 --out runs/deberta_s0
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

DATA = Path(__file__).resolve().parent / "data" / "raw" / "modelling_en_hb5.jsonl"


def load(path):
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    labels = sorted({r["label"] for r in rows})
    return rows, labels


def batches(rows, tok, label_id, max_len, batch, shuffle, seed):
    def collate(chunk):
        enc = tok([r["text"] for r in chunk], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([label_id[r["label"]] for r in chunk])
        return enc
    g = torch.Generator().manual_seed(seed)
    return DataLoader(rows, batch_size=batch, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    out, loss_sum, n = [], 0.0, 0
    for enc in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(**enc).logits.float()
        loss_sum += torch.nn.functional.cross_entropy(logits, enc["labels"], reduction="sum").item()
        n += len(enc["labels"])
        out.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(out), loss_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="train on the first N units (smoke test only)")
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, labels = load(Path(args.data))
    label_id = {c: i for i, c in enumerate(labels)}
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "calib", "test")}
    if args.limit:
        split["train"] = split["train"][: args.limit]
        for s in ("dev", "calib", "test"):
            split[s] = split[s][: args.limit // 4]

    tok = AutoTokenizer.from_pretrained(args.model)
    # transformers 5 keeps the checkpoint's dtype (fp16 for DeBERTa-v3); mixed
    # precision needs fp32 master weights, with fp16 only inside autocast
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels)).float().to(device)
    train = batches(split["train"], tok, label_id, args.max_len, args.batch, True, args.seed)
    evals = {s: batches(split[s], tok, label_id, args.max_len, 64, False, 0) for s in ("dev", "calib", "test")}

    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm"))]
    no_decay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps = args.epochs * len(train)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=amp)

    log = {"args": vars(args), "device": str(device), "labels": labels,
           "split_units": {s: len(v) for s, v in split.items()}, "epochs": []}
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t = time.time()
        for step, enc in enumerate(train, 1):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = model(**enc).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(train)} loss {loss.item():.3f} "
                      f"({(time.time() - t) / step:.2f}s/step)", flush=True)
        dev_proba, dev_loss = predict(model, evals["dev"], device, amp)
        log["epochs"].append({"epoch": epoch, "dev_log_loss": dev_loss, "seconds": time.time() - t})
        print(f"epoch {epoch}: dev log-loss {dev_loss:.4f}", flush=True)
        if best is None or dev_loss < best:
            best = dev_loss
            log["best_epoch"] = epoch
            saved = {"dev": dev_proba}
            for s in ("calib", "test"):
                saved[s], _ = predict(model, evals[s], device, amp)
            np.savez_compressed(
                out / "probs.npz", labels=np.array(labels),
                **{f"{s}_proba": p for s, p in saved.items()},
                **{f"{s}_ids": np.array([f"{r['manifesto_id']}#{r['pos']}" for r in split[s]])
                   for s in saved})
        (out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"best epoch {log['best_epoch']}, dev log-loss {best:.4f}; wrote {out / 'probs.npz'}")


if __name__ == "__main__":
    main()
