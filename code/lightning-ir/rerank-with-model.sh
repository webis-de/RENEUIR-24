#!/usr/bin/bash

lightning-ir re_rank \
	--config config/tinybert-inference.yaml \
	--data.inference_datasets.init_args.run_path msmarco-passage-trec-dl-2019-judged-20230107-training-rerank.jsonl.gz \
	--trainer.callbacks.init_args.save_dir runs/${1}/ \
	--model.init_args.model_name_or_path lightning_logs/${1}/huggingface_checkpoint/

