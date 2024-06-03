import pyterrier as pt
from os.path import abspath
import pandas as pd

if not pt.started():
    pt.init()
    index = pt.IndexRef.of(abspath('index'))
    index = pt.IndexFactory.of(index)
    tokeniser = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()

def pt_tokenise(text):
    return ' '.join(tokeniser.getTokens(text))

def bm25_score(query, docnos):
    global index
    pipeline = pt.FeaturesBatchRetrieve(index, wmodel="TF", features=["WMODEL:BM25", "WMODEL:TF", "WMODEL:TF-IDF", "WMODEL:PL2", "WMODEL:DirichletLM"])

    df = []
    for docno in docnos:
        df += [{
            'qid': '1', 'query': pt_tokenise(query),
            'docno': docno,
        }]

    return None

if __name__ == '__main__':
    #export IR_DATASETS_HOME="/mnt/ceph/storage/data-tmp/current/kibi9872/.ir_datasets"
    bm25_score('hello world', ['7694473', '4373328'])

