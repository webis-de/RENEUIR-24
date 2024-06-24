#!/usr/bin/bash

lightning-ir re_rank \
	--config ${CONFIG_DIR}/tinybert-inference.yaml \
	--data.inference_datasets.init_args.run_path ${inputDataset}/rerank.jsonl.gz \
	--trainer.callbacks.init_args.save_dir ${outputDir}/ \
	--model.init_args.model_name_or_path ${MODEL}

mv ${outputDir}/rerank.run ${outputDir}/run.txt

