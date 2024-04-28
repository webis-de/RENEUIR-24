#!/usr/bin/env bash
##
# Example Draco job script.
##
#SBATCH --job-name=ir_project_training
#SBATCH --output=ir_project_training.%j.out
#SBATCH --error=ir_project_training.%j.err
#SBATCH -p short
#SBATCH -N 1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=96
#SBATCH --time=16:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=

echo "submit host:"
echo $SLURM_SUBMIT_HOST
echo "submit dir:"
echo $SLURM_SUBMIT_DIR
echo "nodelist:"
echo $SLURM_JOB_NODELIST

# activate conda environment
# module load tools/anaconda3/2021.05
# source "$(conda info -a | grep CONDA_ROOT | awk -F ' ' '{print $2}')"/etc/profile.d/conda.sh
# conda activate pytorch_x86

# train MLP
cd $HOME/RENEUIR-24/code
export PYTHONUNBUFFERED=TRUE
python cross-encoder.py