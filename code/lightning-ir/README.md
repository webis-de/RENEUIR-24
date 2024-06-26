docker build -t tiny-lightning-ir .

```
docker run --rm -ti \
    --network=none \
    --entrypoint bash \
    -v ${PWD}/spot-check-dataset/:/input-data:ro \
    -v ${PWD}/output-runs/:/runs \
    -v ${PWD}/lightning_logs/version_9/huggingface_checkpoint/:/model \
    -e MODEL=/model \
    tiny-lightning-ir
```

```
/rerank-with-model.sh /input-data /runs
```

tira-run \
	--input-dataset reneuir-2024/re-rank-spot-check-20240624-training
	--image tiny-lightning-ir

```
docker run --rm -ti \
    --network=host \
    --entrypoint jupyter \
    -v ${PWD}/spot-check-dataset/:/input-data:ro \
    -v ${PWD}:/app \
    -v /mnt/ceph/tira/state/ir_datasets/:/root/.ir_datasets:ro \
    -w /app \
    mam10eks/reneuir-tinybert:0.0.1 \
    notebook --ip 0.0.0.0 --allow-root
```
