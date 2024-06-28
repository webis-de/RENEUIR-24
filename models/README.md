First, run:

```
./models/build-all.sh
```

then, run for each model:

```
MODEL=X docker run --rm -ti \
	    --network=none \
	    -v ${PWD}/code/lightning-ir/msmarco-dev/:/input-data:ro \
	    -v ${PWD}/models/runs/${MODEL}:/runs \
	    -e CUDA_VISIBLE_DEVICES=0 \
	    -e TIRA_INPUT_DATASET=/input-data \
	    -e TIRA_OUTPUT_DIR=/runs \
	    ${MODEL}
```

