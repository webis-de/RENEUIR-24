"""
This examples show how to train a Cross-Encoder for the MS Marco dataset
(https://github.com/microsoft/MSMARCO-Passage-Ranking).

In this example we use a knowledge distillation setup. Sebastian Hofstätter et al. trained in
https://arxiv.org/abs/2010.02666 an ensemble of large Transformer models for the MS MARCO datasets
and combines the scores from a BERT-base, BERT-large, and ALBERT-large model.

We use the logits scores from the ensemble to train a smaller model. We found that the MiniLM model gives the best
performance while offering the highest speed.

The resulting Cross-Encoder can then be used for passage re-ranking: You retrieve for example 100 passages
for a given query, for example with ElasticSearch, and pass the query+retrieved_passage to the CrossEncoder
for scoring. You sort the results then according to the output of the CrossEncoder.

This gives a significant boost compared to out-of-the-box ElasticSearch / BM25 ranking.

Running this script:
python train_cross-encoder-v2.py
"""
from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder

from sentence_transformers import InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import torch
import json
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# /print debug information to stdout

# We make a directory for storing the MS Marco dataset
data_path = '../data/triples-ms-marco-tiny-with-bm25.jsonl.gz'

raw_input = pd.read_json(data_path, encoding="utf-8", lines=True)

# Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}

# Read the train queries, store in queries dict
queries = {}

# Loading injection scores and applying normalization (global min-max in the paper)
global_min_bm25 = 0
global_max_bm25 = 50
scores = {}

train_samples = []

for _, i in raw_input.iterrows():
    corpus[i["id_positive_document"]] = i["positive_document"]
    corpus[i["id_negative_document"]] = i["negative_passage"]

    qid = str(i["id_positive_document"]) + str(i["id_negative_document"])
    queries[qid] = i["query"]

    score_pos = i["score_positive_document"]["BM25"]
    normalized_score_pos = (score_pos - global_min_bm25) / (global_max_bm25 - global_min_bm25)
    normalized_score_pos = int(normalized_score_pos * 100)
    scores[qid, i["id_positive_document"]] = normalized_score_pos

    score_neg = i["score_negative_document"]["BM25"]
    normalized_score_neg = (score_neg - global_min_bm25) / (global_max_bm25 - global_min_bm25)
    normalized_score_neg = int(normalized_score_neg * 100)
    scores[qid, i["id_negative_document"]] = normalized_score_neg

    # Read our training file
    # As input examples, we provide the (query, passage) pair together with the logits score from the teacher ensemble
    train_samples.append(InputExample(texts=["{} [SEP] {}".format(scores[qid, i["id_positive_document"]], queries[qid]),
                                             corpus[i["id_positive_document"]]], label=float(normalized_score_pos)))
    train_samples.append(InputExample(texts=["{} [SEP] {}".format(scores[qid, i["id_negative_document"]], queries[qid]),
                                             corpus[i["id_negative_document"]]], label=float(normalized_score_neg)))


# First, we define the transformer model we want to fine-tune
model_name = "prajjwal1/bert-tiny"
train_batch_size = 32
num_epochs = 10

# We set num_labels=1 and set the activation function to Identiy, so that we get the raw logits
model = CrossEncoder(model_name, num_labels=1, default_activation_function=torch.nn.Identity())

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)

# Configure the training
loss_function = torch.nn.BCEWithLogitsLoss()

# Train the model
model.fit(train_dataloader=train_dataloader,
          loss_fct=loss_function,
          epochs=num_epochs,
          evaluation_steps=1,
          warmup_steps=0,
          optimizer_params={'lr': 7e-6},
          use_amp=True)

# Save latest model
model.save("bm25/cross-encoder-bm25")
