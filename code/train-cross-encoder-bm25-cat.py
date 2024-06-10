"""
This code is copied from https://github.com/arian-askari/injecting_bm25_score_bert_reranker and only adapted to run in a local slurm setup. I.e., data mounted to different locations.
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

from CERerankingEvaluator_bm25cat import CERerankingEvaluator
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

### We make a directory for storing the MS Marco dataset
data_folder = '/mnt/ceph/storage/data-in-progress/data-research/web-search/RENEUIR-24/training-data-bm25-cat/'
injection_folder = data_folder + "/injection_scores/"
os.makedirs(injection_folder, exist_ok=True)

# We download injection scores

train_scores_path = os.path.join(injection_folder, '1_bm25_scores_train_triples_small.json')

validation_scores_path = os.path.join(injection_folder, '5_bm25_scores_train-eval_triples.json')

# Loading injection scores and applying normalization (global min-max in the paper)
global_min_bm25 = 0
global_max_bm25 = 50
scores_path = injection_folder + "1_bm25_scores_train_triples_small.json"
scores = json.loads(open(scores_path, "r").read())
for qid in tqdm.tqdm(scores.keys(), desc = "reading scores...{}".format(scores_path)):
  for did, score in scores[qid].items():
    normalized_score = (score - global_min_bm25) / (global_max_bm25 - global_min_bm25)
    normalized_score = int(normalized_score * 100)
    scores[qid][did] = normalized_score

validation_scores_path = injection_folder + "/5_bm25_scores_train-eval_triples.json"
scores_validation = json.loads(open(validation_scores_path, "r").read())
for qid in tqdm.tqdm(scores_validation.keys(), desc = "reading validation scores...{}".format(validation_scores_path)):
  if qid not in scores:
    scores[qid] = {}
  for did, score in scores_validation[qid].items():
    normalized_score = (score - global_min_bm25) / (global_max_bm25 - global_min_bm25)
    normalized_score = int(normalized_score * 100)
    scores[qid][did] = normalized_score

#First, we define the transformer model we want to fine-tune
model_name = 'microsoft/MiniLM-L12-H384-uncased'
train_batch_size = 32
num_epochs = 1
model_save_path = data_folder + '/../models-bm25-cat/train-cross-encoder-kd-bm25cat-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



#We set num_labels=1 and set the activation function to Identiy, so that we get the raw logits
model = CrossEncoder(model_name, num_labels=1, max_length=512, default_activation_function=torch.nn.Identity())

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query


### Now we create our  dev data
train_samples = []
dev_samples = {}

# We use 200 random queries from the train set for evaluation during training
# Each query has at least one relevant and up to 200 irrelevant (negative) passages
num_dev_queries = 200
num_max_dev_negatives = 200

# msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz and msmarco-qidpidtriples.rnd-shuf.train.tsv.gz is a randomly
# shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website
# We extracted in the train-eval split 500 random queries that can be used for evaluation during training
train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')

with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': list(), 'positive': list(), 'negative': list()}

        if qid in dev_samples:
            dev_samples[qid]['positive'].append(corpus[pos_id])
            dev_samples[qid]['query'].append("{} [SEP] {}".format(scores[qid][pos_id], queries[qid]))

            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].append(corpus[neg_id])
                dev_samples[qid]['query'].append("{} [SEP] {}".format(scores[qid][neg_id], queries[qid]))


dev_qids = set(dev_samples.keys())

# Read our training file
# As input examples, we provide the (query, passage) pair together with the logits score from the teacher ensemble
teacher_logits_filepath = os.path.join(data_folder, 'bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv')
train_samples = []

with open(teacher_logits_filepath) as fIn:
    for line in fIn:
        pos_score, neg_score, qid, pid1, pid2 = line.strip().split("\t")

        if qid in dev_qids: #Skip queries in our dev dataset
            continue

        train_samples.append(InputExample(texts=["{} [SEP] {}".format(scores[qid][pid1], queries[qid]), corpus[pid1]], label=float(pos_score)))
        train_samples.append(InputExample(texts=["{} [SEP] {}".format(scores[qid][pid2], queries[qid]), corpus[pid2]], label=float(neg_score)))

# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

# Configure the training
warmup_steps = 5000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_dataloader=train_dataloader,
          loss_fct=torch.nn.MSELoss(),#
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=5000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr': 7e-6},
          use_amp=True)

#Save latest model
model.save(model_save_path+'-latest')
