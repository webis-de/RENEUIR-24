#!/usr/bin/env bash
##
# Example Draco job script.
##
#SBATCH --job-name=ir_project_training
#SBATCH --gres=gpu:ampere
#SBATCH --output=slurm-output/ir_project_training.%j.out
#SBATCH --error=slurm-output/ir_project_training.%j.err
#SBATCH --container-image=bash
#SBATCH --array=1-10
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
echo "hallo welt ${SLURM_ARRAY_TASK_ID}"
