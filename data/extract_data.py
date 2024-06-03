#!/usr/bin/env python3
from tqdm import tqdm
import pandas as pd
from bm25_scoring import bm25_score
from extract_rank_distill_llm import queries
import json


def build_doc_to_docno(file_name):
    ret = {}
    with open(file_name, 'r') as f:
        for i in tqdm(f, 'Document Index'):
            i = json.loads(i)
            ret[i['text'].lower().strip()] = i['docno']

    return ret


def parse_tsv(file_name, doc_to_docno):
    with open(file_name, 'r') as f:
        covered_queries = set()
        for doc in tqdm(f, 'triples'):
            query, positive_passage, negative_passage = doc.split('\t')
            if query in covered_queries:
                continue
            positive_id = doc_to_docno.get(positive_passage.lower().strip())
            negative_id = doc_to_docno.get(negative_passage.lower().strip())
            if positive_id is None or negative_id is None:
                continue
            covered_queries.add(query)
            yield {'query': query, 'positive_document': positive_passage, 'negative_passage': negative_passage, 'id_positive_document': positive_id, 'id_negative_document': negative_id}

        print(f'Processed {len(covered_queries)} queries')


if __name__ == '__main__':
    doc_to_docno = build_doc_to_docno('documents.jsonl')
    df = list(parse_tsv('triples.train.small.tsv', doc_to_docno))
    print(len(df))
    pd.DataFrame(df).to_json('triples-train.jsonl.gz', lines=True, orient='records')
