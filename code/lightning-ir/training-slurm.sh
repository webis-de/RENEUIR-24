#!/usr/bin/env bash
#SBATCH --job-name=reneuir_distillation_tinybert
#SBATCH --gres=gpu:ampere
#SBATCH --output=slurm-output/reneuir_distillation_tinybert.%j.out
#SBATCH --error=slurm-output/reneuir_distillation_tinybert.%j.err
#SBATCH --container-image=mam10eks/reneuir-tinybert:0.0.1
#SBATCH --array=0-0
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

#MODELS=("/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/models--prajjwal1--bert-tiny/snapshots/6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837" "/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/models--Integer-Ctrl--cross-encoder-bert-tiny-1gb-bs32/snapshots/94de6cd2d169d5a0b15bb17021713ac1cb91e0ed" "/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/models--Integer-Ctrl--cross-encoder-bert-tiny-512/snapshots/7415399968e780d904e7e9a7b84d20304b834ef7" "/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/models--Integer-Ctrl--cross-encoder-bert-tiny-5120/snapshots/845964d41cd8b81a2433e74fb9766b7c9255ec1c" "/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/models--Integer-Ctrl--cross-encoder-bert-tiny-51200/snapshots/3a588aa4b9fbd2e22f33cfb0095e837f38934364" "/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/models--cross-encoder--ms-marco-TinyBERT-L-2/snapshots/e9ed04745b2b19e8c4499360253ea5d5b41b5810")
MODELS=("/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/models--cross-encoder--ms-marco-TinyBERT-L-2/snapshots/e9ed04745b2b19e8c4499360253ea5d5b41b5810")
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
echo "Run task: ${SLURM_ARRAY_TASK_ID} to train model '${MODEL}'"
export IR_DATASETS_HOME='/mnt/ceph/storage/data-tmp/current/fschlatt/.ir_datasets/'
./fit-model.sh ${MODEL}
