import pickle
from typing import List

def build_bm25(texts: List[str]):
    from rank_bm25 import BM25Okapi
    import re
    tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
    return BM25Okapi(tokenized)

def build_tfidf(texts: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=100000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return {"vectorizer": vec, "X": X}

def save_sparse_index(obj, out_path):
    with open(out_path, "wb") as f:
        pickle.dump(obj, f)
