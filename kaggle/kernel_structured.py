"""Kaggle job for step 6: one structured variant, seeds 0, 1 and 2, one seed
per GPU at a time. VARIANT is set in the copy that gets pushed ("hier" or
"defs"), because Kaggle runs this without arguments.

Needs in the private dataset: train_encoder.py, train_structured.py,
modelling_en_hb5.jsonl, codeframe_hb5_structure.csv, and for defs
codeframe_hb5.csv (handbook definitions).
"""

import glob
import os
import subprocess
import sys

import torch

VARIANT = "hier"
SMOKE = False
LANG = "en"                       # "en": DeBERTa v3 base; "nl": mDeBERTa v3 base
MODEL = "microsoft/deberta-v3-base" if LANG == "en" else "microsoft/mdeberta-v3-base"


def find(name):
    return glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]


script = find("train_structured.py")
extra = ["--model", MODEL, "--data", find(f"modelling_{LANG}_hb5.jsonl"),
         "--structure", find("codeframe_hb5_structure.csv")]
if VARIANT == "defs":
    extra += ["--definitions", find("codeframe_hb5.csv")]

seeds = [0] if SMOKE else [0, 1, 2]
gpus = max(1, torch.cuda.device_count())
print("cuda:", torch.cuda.is_available(), "gpus:", gpus, flush=True)
failed = []
for start in range(0, len(seeds), gpus):
    procs = []
    for gpu, seed in enumerate(seeds[start : start + gpus]):
        out = f"/kaggle/working/runs/{LANG}_{VARIANT}_s{seed}"
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, script, "--variant", VARIANT, "--seed", str(seed), "--out", out] + extra
        if SMOKE:
            cmd += ["--limit", "2000", "--epochs", "1"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = open(f"{out}/train.log", "w")
        procs.append((seed, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    for seed, proc, log in procs:
        code = proc.wait()
        log.close()
        print(f"{VARIANT} seed {seed} exited with {code}", flush=True)
        if code:
            failed.append(seed)
if failed:
    raise SystemExit(f"failed seeds: {failed}")
