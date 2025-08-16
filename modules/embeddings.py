from typing import List
import numpy as np

def _load_st_model(name: str, device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, device=device)
    return model

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
    try:
        model = _load_st_model(model_name, device=device)
        vecs = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
        return np.array(vecs, dtype="float32")
    except Exception as e:
        # Fallback: hashing-based cheap embeddings (not great, but avoids crash offline)
        import hashlib
        def hvec(t):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            return arr[:32] / 255.0
        return np.vstack([hvec(t) for t in texts]).astype("float32")

def save_faiss_index(vecs: np.ndarray, ids: list, out_path, metric: str = "ip"):
    try:
        import faiss
    except Exception as e:
        raise RuntimeError("faiss not installed")
    dim = vecs.shape[1]
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add_with_ids(vecs, np.array(ids, dtype=np.int64))
    faiss.write_index(index, str(out_path))
