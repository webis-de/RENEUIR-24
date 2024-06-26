#!/usr/bin/bash

#for MODEL in $(find -L -iname huggingface_checkpoint|awk -F '/' '{print $3}'|sort -u)
for MODEL in "version_26" "version_30"
do
	echo "Run predictions with ${MODEL}"
	docker run --rm -ti \
	    --network=none \
	    -v ${PWD}/msmarco-dev/:/input-data:ro \
	    -v ${PWD}/output-runs/${MODEL}:/runs \
	    -v ${PWD}/lightning_logs/${MODEL}/huggingface_checkpoint/:/model \
	    -e CUDA_VISIBLE_DEVICES=2 \
	    -e MODEL=/model \
	    -e inputDataset=/input-data \
	    -e outputDir=/runs \
	    tiny-lightning-ir
done
