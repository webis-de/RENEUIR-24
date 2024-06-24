FROM mam10eks/reneuir-tinybert:0.0.1

ADD rerank-with-model.sh /rerank-with-model.sh
ADD config /config

ENV CONFIG_DIR=/config


ENTRYPOINT [ "/rerank-with-model.sh" ]

