#!/bin/bash
#SBATCH --job-name=toy_trl
#SBATCH --partition=hive
#SBATCH --qos=hive_intermediate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

export MAMBA_ROOT_PREFIX=/scratch_hive/$USER/micromamba-root
eval "$(/scratch_hive/$USER/micromamba-bin/bin/micromamba shell hook -s bash)"
micromamba activate localpaperreader

export HF_HUB_OFFLINE=1
cd /scratch_hive/kw1525/LocalPaperReader/grpo
PYTHONPATH=. python toy/train_trl.py --difficulty 3 --steps 300 --out runs/trl_d3