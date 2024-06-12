#!/usr/bin/env bash
#SBATCH --job-name=reneuir_distillation_tinybert
#SBATCH --gres=gpu:2g.10gb
#SBATCH --output=slurm-output/reneuir_distillation_tinybert.%j.out
#SBATCH --error=slurm-output/reneuir_distillation_tinybert.%j.err
#SBATCH --container-image=bash
#SBATCH --array=0-4
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00

echo "submit host:"
echo $SLURM_SUBMIT_HOST
echo "submit dir:"
echo $SLURM_SUBMIT_DIR
echo "nodelist:"
echo $SLURM_JOB_NODELIST

# train MLP
cd /mnt/ceph/storage/data-tmp/current/kibi9872/RENEUIR-24/code/lightning-ir
export PYTHONUNBUFFERED=TRUE

MODELS=("cat" "dog" "mouse" "frog")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "Run task: ${SLURM_ARRAY_TASK_ID} to train model ${MODEL}"
#python3 train-cross-encoder-bm25-cat.py
