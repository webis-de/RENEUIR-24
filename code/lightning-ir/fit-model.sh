#!/usr/bin/bash

lightning-ir fit --config config/tinybert.yaml --model.init_args.model_name_or_path ${1}

