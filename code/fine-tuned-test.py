import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from transformers import AutoModel, AutoModelForSequenceClassification
from evaluate import load
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
from sentence_transformers import InputExample
import sentence_transformers

# For a full training on the msmarco dataset we decided to use the
# msmarco-passage_train_triples-small dataset.
# This is already provided by the huggingface datasets library.
# https://huggingface.co/datasets/irds/msmarco-passage_train_triples-small


# https://huggingface.co/docs/datasets/tabular_load#pandas-dataframes
# load Dataset from Pandas DataFrame
raw_input = pd.read_json("../data/triples-ms-marco-tiny.jsonl.gz", encoding="utf-8", lines=True)
df = []

for _, i in raw_input.iterrows():
    df += [
        {'query': i["query"],
         'text': i["positive_document"],
         'label': 1.0},
        {'query': i["query"],
         'text': i["negative_passage"],
         'label': 0.0},
    ]

# checkpoint of the models

# trained on bert-tiny with 512 random entries from the msmarco dataset with batch size 32
checkpoint512 = "Integer-Ctrl/cross-encoder-bert-tiny-512"
# trained on bert-tiny with 5120 random entries from the msmarco dataset with batch size 32
checkpoint5120 = "Integer-Ctrl/cross-encoder-bert-tiny-5120"
# trained on bert-tiny with 51200 random entries from the msmarco dataset with batch size 32
checkpoint51200 = "Integer-Ctrl/cross-encoder-bert-tiny-51200"
# trained on bert-tiny with the full msmarco dataset with batch size 200
checkpoint = "Integer-Ctrl/cross-encoder-bert-tiny-bs-200"


# We set num_labels=1 and set the activation function to Identity, so that we get the raw logits
# if cuda is available, it will be automatically used
model512 = CrossEncoder(checkpoint512, num_labels=1)
model5120 = CrossEncoder(checkpoint5120, num_labels=1)
model51200 = CrossEncoder(checkpoint51200, num_labels=1)
model = CrossEncoder(checkpoint, num_labels=1)

print("Label[1337]: ", df[1337]["label"])
print("Label[420]: ", df[420]["label"])

prediction512_1337 = model512.predict([df[1337]["query"], df[1337]["text"]])
print("Prediction on label 1337 of model 512: ", prediction512_1337)
prediction512_420 = model512.predict([df[420]["query"], df[420]["text"]])
print("Prediction on label 420 of model 512: ", prediction512_420)

prediction5120_1337 = model5120.predict([df[1337]["query"], df[1337]["text"]])
print("Prediction on label 1337 of model 5120: ", prediction5120_1337)
prediction5120_420 = model5120.predict([df[420]["query"], df[420]["text"]])
print("Prediction on label 420 of model 5120: ", prediction5120_420)

prediction51200_1337 = model51200.predict([df[1337]["query"], df[1337]["text"]])
print("Prediction on label 1337 of model 51200: ", prediction51200_1337)
prediction51200_420 = model51200.predict([df[420]["query"], df[420]["text"]])
print("Prediction on label 420 of model 51200: ", prediction51200_420)

prediction_1337 = model.predict([df[1337]["query"], df[1337]["text"]])
print("Prediction on label 1337 of model: ", prediction_1337)
prediction_420 = model.predict([df[420]["query"], df[420]["text"]])
print("Prediction on label 420 of model: ", prediction_420)
