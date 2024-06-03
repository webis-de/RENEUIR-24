#!/usr/bin/env python3
from tqdm import tqdm
import pandas as pd


def parse_tsv(file_name):
    with open(file_name, 'r') as f:
        covered_queries = set()
        for doc in tqdm(f):
            query, positive_passage, negative_passage = doc.split('\t')
            if query in covered_queries:
                continue

            covered_queries.add(query)
            yield {'query': query, 'positive_document': positive_passage, 'negative_passage': negative_passage}

        print(f'Processed {len(covered_queries)} queries')


if __name__ == '__main__':
    df = list(parse_tsv('triples.train.small.tsv'))
    print(len(df))
    pd.DataFrame(df).to_json('triples-train.jsonl.gz', lines=True, orient='records')
