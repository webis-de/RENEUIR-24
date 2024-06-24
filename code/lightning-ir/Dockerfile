FROM mam10eks/reneuir-tinybert:0.0.1

ADD rerank-with-model.sh /rerank-with-model.sh

ENTRYPOINT [ "rerank-with-model.sh", "$inputDataset", "$outputDir" ]

