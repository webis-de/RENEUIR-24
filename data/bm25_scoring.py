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
    pipeline = pt.FeaturesBatchRetrieve(index, wmodel="None", features=["WMODEL:BM25", "WMODEL:TF_IDF", "WMODEL:DirichletLM", "WMODEL:PL2"])
    pipeline.wmodel = None

    df = []
    for docno in docnos:
        df += [{'qid': '1', 'query': pt_tokenise(query), 'docno': docno, 'score': 10000 - len(df)}]
    df = pd.DataFrame(df)

    ret = {}
    for _, i in pipeline(df).iterrows():
        ret[i['docno']] = {'BM25': i['features'][0], 'TF_IDF': i['features'][1], 'DirichletLM': i['features'][2], 'PL2': i['features'][3]}

    return ret

if __name__ == '__main__':
    #export IR_DATASETS_HOME="/mnt/ceph/storage/data-tmp/current/kibi9872/.ir_datasets"
    print(bm25_score('hello world', ['7694473', '4373328']))

