FROM mam10eks/reneuir-tinybert:0.0.1

ADD tiny-bert-ranker-distillation/rerank-with-model.sh /rerank-with-model.sh
ADD tiny-bert-ranker-distillation/config /config

ARG MODEL_NAME
RUN huggingface-cli download $MODEL_NAME

ENV MODEL=/model
ENV CONFIG_DIR=/config


ENTRYPOINT [ "/rerank-with-model.sh" ]

