# code

All code fragments and helper functions are located in this directory.

## cross-encoder.py

Trains a CrossEncoder on the dataset _triples-ms-marco-tiny.jsonl.gz_ using the existing fitting method **fit**. The trained model is stored in _/data/cross-encoder/_.

**Before starting, adjust the code arguments to suit your needs, e.g. batch size, epochs.**

## cross-encoder-custom.py

Trains a CrossEncoder on the dataset _triples-ms-marco-tiny.jsonl.gz_ using a custom training loop. The trained model will be saved in _/data/cross-encoder-custom/_.

**Before starting, adjust the code arguments to suit your needs, e.g. batch size, epochs.**

## training-slurm.sh

Slurm script to start a training using the _cross-encoder.py_ trainer.

**Adjust the slurm arguments to your needs and server architecture before starting.**

Start script: `sbatch training-slurm.sh` \
Monitor job: `squeue - u <username>` \
Canceling job: `scancel <job_id>`