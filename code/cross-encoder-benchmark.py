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


# The model we want to fine-tune
checkpoint = "prajjwal1/bert-tiny"

# We define our loss funtion
loss_function = torch.nn.BCEWithLogitsLoss()


def train(train_batch_size, num_epoch, learning_rate, random_seed, checkpoint_name):
    # set random seed
    torch.manual_seed(random_seed)
    # We set num_labels=1 and set the activation function to Identity, so that we get the raw logits
    # if cuda is available, it will be automatically used
    model = CrossEncoder(checkpoint, num_labels=1)
    # We create a DataLoader to load our train samples
    train_dataloader = DataLoader(df, shuffle=True, batch_size=train_batch_size)
    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        loss_fct=loss_function,
        epochs=num_epoch,
        evaluation_steps=1,
        warmup_steps=0,
        optimizer_params={"lr": learning_rate},
    )
    # Save latest model
    model.save(checkpoint_name)


# configs
baseline_batch_size = 200
baseline_num_epochs = 10
baseline_learning_rate = 2e-5
baseline_random_seed = 233

train_batch_sizes = [16, 64, 128]
train_num_epochs = [5, 15, 20]
train_learning_rates = [1e-5, 1e-4, 5e-4]
train_random_seeds = [5, 6765, 46368]  # 05/13/2024

# train baseline model
train(baseline_batch_size,
      baseline_num_epochs,
      baseline_learning_rate,
      baseline_random_seed,
      "cross-encoder-32-10-2e5-233")

# train hyperparameter search
for batch_size in train_batch_sizes:
    train(batch_size,
          baseline_num_epochs,
          baseline_learning_rate,
          baseline_random_seed,
          f"cross-encoder-{batch_size}-10-2e5-42")

for num_epochs in train_num_epochs:
    train(baseline_batch_size,
          num_epochs,
          baseline_learning_rate,
          baseline_random_seed,
          f"cross-encoder-32-{num_epochs}-2e5-42")

for learning_rate in train_learning_rates:
    train(baseline_batch_size,
          baseline_num_epochs,
          learning_rate,
          baseline_random_seed,
          f"cross-encoder-32-10-{learning_rate}-42")

for random_seed in train_random_seeds:
    train(baseline_batch_size,
          baseline_num_epochs,
          baseline_learning_rate,
          random_seed,
          f"cross-encoder-32-10-2e5-{random_seed}")
