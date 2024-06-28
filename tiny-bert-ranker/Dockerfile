# docker build -t tinybert-plain:0.0.1 -f Dockerfile.tinybert .
FROM mam10eks/reneuir-tinybert:0.0.1

ADD tiny-bert-ranker/run-notebook.py /run-notebook.py
ADD tiny-bert-ranker/re-rank-with-tiny-bert.ipynb /re-rank-with-tiny-bert.ipynb
RUN pip3 uninstall -y tira && pip3 install runnb tira && jupyter trust /re-rank-with-tiny-bert.ipynb

ARG MODEL_NAME
ENV MODEL=$MODEL_NAME
RUN huggingface-cli download $MODEL && python3 -c 'from sentence_transformers import CrossEncoder; import os; CrossEncoder(os.environ["MODEL"])'

ENTRYPOINT [ "/run-notebook.py", "--notebook", "/re-rank-with-tiny-bert.ipynb" ]

