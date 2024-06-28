# docker build -t mam10eks/reneuir-tinybert:0.0.1 .
FROM fschlatt/slurm:0.0.1

RUN pip3 install tira sentence-transformers

RUN git clone https://github.com/webis-de/lightning-ir.git /lightning-ir \
	&& cd /lightning-ir \
	&& pip3 install .

