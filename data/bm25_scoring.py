import pyterrier as pt
from os.path import abspath

if not pt.started():
    pt.init()
    index = pt.IndexRef.of(abspath('index'))
    index = pt.IndexFactory.of(index)

def bm25_score(qid, docnos):
    global index

    return None

