"""Kaggle job for step 5: runs train_encoder.py for seeds 0, 1 and 2.

Kaggle mounts the private dataset (train_encoder.py plus the modelling set)
read-only under /kaggle/input/<dataset>/; outputs go to /kaggle/working/runs/
and are fetched with `kaggle kernels output`. Kaggle runs this without
arguments, so SMOKE is a constant: the smoke copy pushed first sets it to True
for a few-minute check that the environment works before spending GPU hours.
"""

import glob
import subprocess
import sys

SMOKE = False
script = glob.glob("/kaggle/input/**/train_encoder.py", recursive=True)[0]
data = glob.glob("/kaggle/input/**/modelling_en_hb5.jsonl", recursive=True)[0]

subprocess.run([sys.executable, "-c",
                "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
               check=True)
import os

import torch

# one seed per GPU at a time (Kaggle gives two T4s); each seed's log goes to its run folder
seeds = [0] if SMOKE else [0, 1, 2]
gpus = max(1, torch.cuda.device_count())
failed = []
for start in range(0, len(seeds), gpus):
    procs = []
    for gpu, seed in enumerate(seeds[start : start + gpus]):
        out = f"/kaggle/working/runs/deberta_s{seed}"
        os.makedirs(out, exist_ok=True)
        cmd = [sys.executable, script, "--data", data, "--seed", str(seed), "--out", out]
        if SMOKE:
            cmd += ["--limit", "2000", "--epochs", "1"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = open(f"{out}/train.log", "w")
        procs.append((seed, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    for seed, proc, log in procs:
        code = proc.wait()
        log.close()
        print(f"seed {seed} exited with {code}", flush=True)
        if code:
            failed.append(seed)
if failed:
    raise SystemExit(f"failed seeds: {failed}")
