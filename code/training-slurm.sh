#!/usr/bin/env bash
#SBATCH --job-name=ir_project_training
#SBATCH --gres=gpu:ampere
#SBATCH --output=slurm-output/ir_project_training.%j.out
#SBATCH --error=slurm-output/ir_project_training.%j.err
#SBATCH --container-image=mam10eks/reneuir-tinybert:0.0.1
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
cd $PWD/RENEUIR-24/code
export PYTHONUNBUFFERED=TRUE
echo "Run task: ${SLURM_ARRAY_TASK_ID}"
python3 train-cross-encoder-bm25-cat-ms-marco-TinyBERT-L-2.py
# python3 train-cross-encoder-bm25-cat-early-stopping-ms-marco-TinyBERT-L-2.py