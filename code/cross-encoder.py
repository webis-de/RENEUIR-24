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
        InputExample(texts=[i["query"], i["positive_document"]], label=1),
    ]
    df += [
        InputExample(texts=[i["query"], i["negative_passage"]], label=0),
    ]

# training_data = Dataset.from_pandas(pd.DataFrame(df))

# print(training_data)
# print(training_data[0])

# The model we want to fine-tune
checkpoint = "prajjwal1/bert-tiny"
train_batch_size = 200
num_epochs = 10

# We set num_labels=1 and set the activation function to Identity, so that we get the raw logits
# if cuda is available, it will be automatically used
model = CrossEncoder(checkpoint, num_labels=1)

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(df, shuffle=True, batch_size=train_batch_size)

# We define our loss funtion
loss_function = torch.nn.BCEWithLogitsLoss()

# Train the model
model.fit(
    train_dataloader=train_dataloader,
    loss_fct=loss_function,
    epochs=num_epochs,
    evaluation_steps=1,
    warmup_steps=0,
)


# Save latest model
model.save("cross-encoder-bert-tiny-bs-200")
