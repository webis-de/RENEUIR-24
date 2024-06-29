FROM webis/ir-lab-wise-2023:0.0.4

ADD tiny-bert-ranker/run-notebook.py /run-notebook.py
ADD foo/ir-lab-rank-fusion.ipynb /ir-lab-rank-fusion.ipynb
RUN pip3 install ranx && jupyter trust /ir-lab-rank-fusion.ipynb

ENTRYPOINT [ "/run-notebook.py", "--notebook", "/ir-lab-rank-fusion.ipynb" ]
