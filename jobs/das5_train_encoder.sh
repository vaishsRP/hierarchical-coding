#!/bin/bash
# Step 5 on DAS-5: fine-tune the strong flat baseline, three seeds in one job.
#
# One-time setup on the cluster (see the course's DAS-5 page):
#   mkdir -p /var/scratch/$USER && cd /var/scratch/$USER
#   install miniconda there, then:
#   conda create -y -n hc python=3.11 && conda activate hc
#   pip install torch --index-url https://download.pytorch.org/whl/cu118
#   pip install "transformers>=4.40" sentencepiece protobuf numpy
#
# Copy only these to /var/scratch/$USER/hierarchical-coding/ :
#   train_encoder.py
#   data/raw/modelling_en_hb5.jsonl   (Manifesto Project text: do not share further)
#
# Queue it outside working hours, as the cluster rules require:
#   sbatch --begin=18:00 jobs/das5_train_encoder.sh
# Check in the morning with: squeue -u $USER, then copy runs/ back.

#SBATCH --job-name=hc-encoder
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=proq
#SBATCH --gres=gpu:1
#SBATCH --output=runs/slurm-%j.out

module load cuda11.7/toolkit 2>/dev/null || module load cuda11.3/toolkit 2>/dev/null || true

source $HOME/.bashrc
conda activate hc
cd /var/scratch/$USER/hierarchical-coding
export HF_HOME=/var/scratch/$USER/hf-cache

python -c "import torch; print('cuda available:', torch.cuda.is_available())"
for seed in 0 1 2; do
    python train_encoder.py --seed $seed --out runs/deberta_s$seed
done
