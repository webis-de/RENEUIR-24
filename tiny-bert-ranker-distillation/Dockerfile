FROM mam10eks/reneuir-tinybert:0.0.1

ADD tiny-bert-ranker-distillation/rerank-with-model.sh /rerank-with-model.sh
ADD tiny-bert-ranker-distillation/config /config

ARG MODEL_NAME
RUN pip3 install --no-deps ir_datasets \
	&& pip3 install lightning ir_measures faiss-cpu jsonargparse \
	&& huggingface-cli download $MODEL_NAME

ENV MODEL=$MODEL_NAME
ENV CONFIG_DIR=/config


ENTRYPOINT [ "/rerank-with-model.sh" ]

