#!/usr/bin/env python3
from tqdm import tqdm
import pandas as pd
from bm25_scoring import bm25_score
import json



if __name__ == '__main__':
    inp = list(i.to_dict() for _, i in pd.read_json('triples-train.jsonl.gz', lines=True).iterrows())
    df = []

    for i in tqdm(inp):
        scores = bm25_score(i['query'], [str(i['id_positive_document']), str(i['id_negative_document'])])
        i['score_negative_document'] = scores[str(i['id_negative_document'])]
        i['score_positive_document'] = scores[str(i['id_positive_document'])]
        df += [i]

    print(len(df))
    pd.DataFrame(df).to_json('triples-train-with-bm25.jsonl.gz', lines=True, orient='records')
