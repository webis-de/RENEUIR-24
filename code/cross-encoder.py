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
import sentence_transformers

# For a full training on the msmarco dataset we decided to use the
# msmarco-passage_train_triples-small dataset.
# This is already provided by the huggingface datasets library.
# https://huggingface.co/datasets/irds/msmarco-passage_train_triples-small


# https://huggingface.co/docs/datasets/tabular_load#pandas-dataframes
# load Dataset from Pandas DataFrame
raw_input = pd.read_json("../data/triples-ms-marco-tiny.jsonl.gz", encoding="utf-8", lines=True)
df = []

for _, i in df.iterrows():
    df += [
        {'query': i["query"],
         'text': i["postive_document"],
         'label': 1},
        {'query': i["query"],
         'text': i["negative_passage"],
         'label': 0}
    ]

training_data = Dataset.from_pandas(pd.DataFrame(df))


# The model we want to fine-tune
checkpoint = "bert-base-uncased"
train_batch_size = 32
num_epochs = 1

# We set num_labels=1 and set the activation function to Identity, so that we get the raw logits
model = CrossEncoder(checkpoint, num_labels=1, max_length=512)

# We create a DataLoader to load our train samples
# train_dataloader = DataLoader(raw_datasets, shuffle=True, batch_size=train_batch_size)

# We define our loss funtion
loss_function = torch.nn.BCEWithLogitsLoss()

# TODO: write training with (fit) method
# TODO: write training loop

'''
# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=10000,
    warmup_steps=warmup_steps,
    use_amp=True,
)
'''

# Save latest model
model.save("cross-encoder-ms-marco-tiny")
