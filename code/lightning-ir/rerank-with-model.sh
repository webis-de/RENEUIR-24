#!/usr/bin/bash

lightning-ir re_rank \
	--config config/tinybert-inference.yaml \
	--data.inference_datasets.init_args.run_path ${1}/rerank.jsonl.gz \
	--trainer.callbacks.init_args.save_dir ${2}/ \
	--model.init_args.model_name_or_path ${MODEL}

