#!/usr/bin/bash

DIR="/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/models-bm25-cat/fine-tuned/"

for MODEL in $(find ${DIR} -name '*cross-encoder*' | awk -F '/' '{print $11}')
do
	echo "Run predictions with ${MODEL}"
	docker run --rm -ti \
	    --network=none \
	    -v ${PWD}/lightning-ir/msmarco-dev/:/input-data:ro \
	    -v ${PWD}/output-runs-tiny-bert-plain/${MODEL}:/runs \
	    -v ${DIR}/${MODEL}/:/model \
	    -e MODEL=/model \
	    -e CUDA_VISIBLE_DEVICES=0 \
	    -e TIRA_INPUT_DATASET=/input-data \
	    -e TIRA_OUTPUT_DIR=/runs \
	    tinybert-plain:0.0.1
done