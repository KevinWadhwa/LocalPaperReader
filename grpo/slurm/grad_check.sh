#!/bin/bash
#SBATCH --job-name=grad_check
#SBATCH --partition=hive
#SBATCH --qos=hive_intermediate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

export MAMBA_ROOT_PREFIX=/scratch_hive/$USER/micromamba-root
eval "$(/scratch_hive/$USER/micromamba-bin/bin/micromamba shell hook -s bash)"
micromamba activate localpaperreader

export HF_HUB_OFFLINE=1
cd /scratch_hive/kw1525/LocalPaperReader/grpo
for bf16 in 1 0; do
    echo "===== bf16=$bf16 ====="
    PYTHONPATH=. python toy/grad_check.py --bf16 $bf16 "$@"
done