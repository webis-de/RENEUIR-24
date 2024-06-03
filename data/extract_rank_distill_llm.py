#!/usr/bin/env python3
from ir_datasets.util import RequestsDownload, Cache, home_path
from ir_datasets.formats import TrecScoredDocs
import ir_datasets
import gzip
from tqdm import tqdm
import json
from bm25_scoring import bm25_score

dataset = ir_datasets.load("msmarco-passage/train")

def scored_docs(rank_distill_llm_run='__rankzephyr-colbert-10000-sampled-100__msmarco-passage-train-judged.run'):
    base_path = home_path() / 'rank-disti-llm'
    requests_download = RequestsDownload(f'https://zenodo.org/records/11147862/files/{rank_distill_llm_run}?download=1')
    scored_docs = TrecScoredDocs(Cache(requests_download, base_path/rank_distill_llm_run))
    
    return scored_docs

def queries():
    return {i.query_id: i.default_text() for i in dataset.queries_iter()}

if __name__ == '__main__':
    q = queries()
    docs_store = dataset.docs_store()
    with gzip.open('rank-distill-llm.jsonl.gz', 'wt') as f:
        for i in tqdm(list(scored_docs().scoreddocs_iter())):
            f.write(json.dumps({'query_id': i.query_id, 'doc_id': i.doc_id, 'score': i.score, 'query': q[i.query_id], 'text': docs_store.get(i.doc_id).default_text()}) + '\n')

