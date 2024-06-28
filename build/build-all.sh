#!/usr/bin/bash -e

docker build -t tiny-bert-ranker -f code/Dockerfile.tinybert --build-arg MODEL_NAME=webis/tiny-bert-ranker .
docker build -t tiny-bert-ranker-l-2 -f code/Dockerfile.tinybert --build-arg MODEL_NAME=cross-encoder/ms-marco-TinyBERT-L-2 .

docker build -t tiny-bert-ranker-bm25 -f tiny-bert-ranker-bm25/Dockerfile  --build-arg MODEL_NAME=webis/tiny-bert-ranker-bm25 .
docker build -t tiny-bert-ranker-l2-bm25 -f tiny-bert-ranker-bm25/Dockerfile  --build-arg MODEL_NAME=webis/tiny-bert-ranker-l2-bm25 .


docker build -t tiny-bert-ranker-distillation -f code/lightning-ir/Dockerfile --build-arg MODEL_NAME=webis/tiny-bert-ranker-distillation .
docker build -t tiny-bert-ranker-l-2-distillation -f code/lightning-ir/Dockerfile --build-arg MODEL_NAME=webis/tiny-bert-ranker-l-2-distillation .

