"""Step 6: structured variants, each against the strong flat baseline.

Same encoder, data, hyperparameters, seeds and epoch selection as
train_encoder.py, so the only thing that changes is how the hierarchy is used.

  --variant hier   hierarchy in the output: p(leaf) = p(domain) * p(leaf | domain),
                   one domain head and one leaf head on a shared encoder, trained
                   jointly on the leaf likelihood. Domain probabilities are saved
                   too, for the strict "domain first" decode.
  --variant defs   hierarchy in the input: a dual encoder scores each sentence
                   against the handbook definition text of every leaf, cosine
                   similarity over a learned temperature.
                   Definitions pass through the same encoder, with gradients.

Needs, next to the modelling set:
  codeframe_hb5_structure.csv   parents (both variants)
  codeframe_hb5.csv             definitions (defs only; handbook text, keep private)

Writes <out>/probs.npz in the same format as train_encoder.py (plus
domain probabilities for hier) and <out>/log.json.

Usage: python train_structured.py --variant hier --seed 0 --out runs/hier_s0
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

import train_encoder as te

HERE = Path(__file__).resolve().parent


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def domain_index(labels, structure_csv):
    parent = {r["code"]: r["parent_code"] for r in read_csv(structure_csv)}

    def domain(code):
        while code and not code.startswith("domain_"):
            code = parent[code]
        return code

    domains = sorted({domain(c) for c in labels})
    return domains, torch.tensor([domains.index(domain(c)) for c in labels])


class Pooler(nn.Module):
    """CLS token, dense, GELU, dropout; the same shape of head the flat model uses."""

    def __init__(self, hidden):
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(0.1)

    def forward(self, last_hidden):
        return self.drop(nn.functional.gelu(self.dense(last_hidden[:, 0])))


class HierModel(nn.Module):
    def __init__(self, name, n_leaves, leaf_domain):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name).float()
        h = self.encoder.config.hidden_size
        self.pool = Pooler(h)
        self.n_domains = int(leaf_domain.max()) + 1
        self.domain_head = nn.Linear(h, self.n_domains)
        self.leaf_head = nn.Linear(h, n_leaves)
        self.register_buffer("leaf_domain", leaf_domain)

    def forward(self, input_ids, attention_mask, **_):
        z = self.pool(self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        log_pd = torch.log_softmax(self.domain_head(z).float(), dim=-1)          # (B, D)
        leaf = self.leaf_head(z).float()                                          # (B, L)
        # softmax of leaf scores within each domain
        log_within = torch.empty_like(leaf)
        for d in range(self.n_domains):
            m = self.leaf_domain == d
            log_within[:, m] = torch.log_softmax(leaf[:, m], dim=-1)
        log_leaf = log_pd[:, self.leaf_domain] + log_within                       # sums to 1 over leaves
        return log_leaf, log_pd


class DefsModel(nn.Module):
    def __init__(self, name, def_batch):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name).float()
        # 63 definitions pass through the encoder with gradients every step; recomputing
        # activations instead of storing them makes this fit on a 16 GB GPU (same maths)
        self.encoder.gradient_checkpointing_enable()
        self.def_batch = def_batch                                                 # tokenised definitions
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.05)))           # temperature 0.05 at start
        self.def_cache = None                                                      # eval only

    def embed(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)          # mean pooling

    def forward(self, input_ids, attention_mask, **_):
        s = self.embed(input_ids, attention_mask)
        if self.training or self.def_cache is None:
            d = self.embed(self.def_batch["input_ids"], self.def_batch["attention_mask"])
        else:
            d = self.def_cache
        logits = (s @ d.T).float() * self.log_scale.exp().clamp(max=100)
        return torch.log_softmax(logits, dim=-1), None


@torch.no_grad()
def predict(model, loader, device, amp):
    model.eval()
    if isinstance(model, DefsModel):
        # definitions do not change during evaluation: encode them once
        model.def_cache = None
        with torch.autocast(device_type=device.type, enabled=amp):
            model.def_cache = model.embed(model.def_batch["input_ids"], model.def_batch["attention_mask"])
    leaf_out, dom_out, loss_sum, n = [], [], 0.0, 0
    for enc in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=device.type, enabled=amp):
            log_leaf, log_pd = model(**enc)
        loss_sum += nn.functional.nll_loss(log_leaf.float(), enc["labels"], reduction="sum").item()
        n += len(enc["labels"])
        leaf_out.append(log_leaf.float().exp().cpu().numpy())
        if log_pd is not None:
            dom_out.append(log_pd.float().exp().cpu().numpy())
    if isinstance(model, DefsModel):
        model.def_cache = None
    return np.vstack(leaf_out), (np.vstack(dom_out) if dom_out else None), loss_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["hier", "defs"], required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--def_len", type=int, default=192)  # longest leaf definition (108) is 164 tokens
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    ap.add_argument("--data", default=str(te.DATA))
    ap.add_argument("--structure", default=str(HERE / "data" / "codeframe_hb5_structure.csv"))
    ap.add_argument("--definitions", default=str(HERE / "data" / "codeframe_hb5.csv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, labels = te.load(Path(args.data))
    label_id = {c: i for i, c in enumerate(labels)}
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "calib", "test")}
    if args.limit:
        split["train"] = split["train"][: args.limit]
        for s in ("dev", "calib", "test"):
            split[s] = split[s][: args.limit // 4]

    tok = AutoTokenizer.from_pretrained(args.model)
    domains, leaf_domain = domain_index(labels, args.structure)
    if args.variant == "hier":
        model = HierModel(args.model, len(labels), leaf_domain)
    else:
        text = {r["code"]: r["definition"] for r in read_csv(args.definitions)}
        defs = [text[c] for c in labels]
        missing = [c for c, d in zip(labels, defs) if not d.strip()]
        if missing:
            raise SystemExit(f"leaves without a handbook definition: {missing}")
        def_batch = tok(defs, truncation=True, max_length=args.def_len, padding=True, return_tensors="pt")
        model = DefsModel(args.model, {k: v.to(device) for k, v in def_batch.items()})
    model.to(device)

    train = te.batches(split["train"], tok, label_id, args.max_len, args.batch, True, args.seed)
    evals = {s: te.batches(split[s], tok, label_id, args.max_len, 64, False, 0) for s in ("dev", "calib", "test")}

    named = list(model.named_parameters())
    decay = [p for n, p in named if not any(k in n for k in ("bias", "LayerNorm", "log_scale"))]
    no_decay = [p for n, p in named if any(k in n for k in ("bias", "LayerNorm", "log_scale"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps = args.epochs * len(train)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=amp)

    log = {"args": vars(args), "device": str(device), "labels": labels, "domains": domains,
           "split_units": {s: len(v) for s, v in split.items()}, "epochs": []}
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t = time.time()
        for step, enc in enumerate(train, 1):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device.type, enabled=amp):
                log_leaf, _ = model(**enc)
            loss = nn.functional.nll_loss(log_leaf.float(), enc["labels"])
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
        dev_leaf, dev_dom, dev_loss = predict(model, evals["dev"], device, amp)
        log["epochs"].append({"epoch": epoch, "dev_log_loss": dev_loss, "seconds": time.time() - t})
        print(f"epoch {epoch}: dev log-loss {dev_loss:.4f}", flush=True)
        if best is None or dev_loss < best:
            best = dev_loss
            log["best_epoch"] = epoch
            saved = {"dev": (dev_leaf, dev_dom)}
            for s in ("calib", "test"):
                leaf_p, dom_p, _ = predict(model, evals[s], device, amp)
                saved[s] = (leaf_p, dom_p)
            arrays = {"labels": np.array(labels), "domains": np.array(domains)}
            for s, (leaf_p, dom_p) in saved.items():
                arrays[f"{s}_proba"] = leaf_p
                arrays[f"{s}_ids"] = np.array([f"{r['manifesto_id']}#{r['pos']}" for r in split[s]])
                if dom_p is not None:
                    arrays[f"{s}_domain_proba"] = dom_p
            np.savez_compressed(out / "probs.npz", **arrays)
        (out / "log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"best epoch {log['best_epoch']}, dev log-loss {best:.4f}; wrote {out / 'probs.npz'}")


if __name__ == "__main__":
    main()
