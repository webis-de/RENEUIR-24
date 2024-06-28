#!/usr/bin/bash

lightning-ir re_rank \
	--config ${CONFIG_DIR}/tinybert-inference.yaml \
	--data.inference_datasets.init_args.run_path ${TIRA_INPUT_DATASET}/rerank.jsonl.gz \
	--trainer.callbacks.init_args.save_dir ${TIRA_OUTPUT_DIR}/ \
	--model.init_args.model_name_or_path ${MODEL}

mv ${TIRA_OUTPUT_DIR}/rerank.run ${TIRA_OUTPUT_DIR}/run.txt

