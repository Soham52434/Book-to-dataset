import os, math
from typing import List, Dict

def _resolve_env(value: str) -> str:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        key = value[2:-1]
        return os.getenv(key, "")
    return value

def _get_cfg(cfg: Dict, key: str, default=None):
    v = cfg.get("vectordb", {}).get(key, default)
    if isinstance(v, str):
        return _resolve_env(v)
    return v

def push_to_pinecone(vecs, chunks: List[Dict], cfg: Dict):
    import numpy as np
    try:
        from pinecone import Pinecone, ServerlessSpec
    except Exception as e:
        raise RuntimeError("pinecone-client not installed. pip install pinecone-client") from e

    api_key = _get_cfg(cfg, "api_key") or os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Pinecone API key missing. Set PINECONE_API_KEY or provide in config as ${PINECONE_API_KEY}.")

    index_name = _get_cfg(cfg, "index_name")
    if not index_name:
        raise ValueError("vectordb.index_name is required for Pinecone.")

    metric = _get_cfg(cfg, "metric", "cosine")
    region = _get_cfg(cfg, "environment", "us-east-1")
    namespace = _get_cfg(cfg, "namespace", None)

    pc = Pinecone(api_key=api_key)

    # Create index if missing
    names = set([i.name for i in pc.list_indexes()])
    dim = int(vecs.shape[1])
    if index_name not in names:
        spec = ServerlessSpec(cloud="aws", region=region)
        pc.create_index(name=index_name, dimension=dim, metric=metric, spec=spec)

    index = pc.Index(index_name)

    # Upsert in batches
    N = vecs.shape[0]
    batch = 200
    for i in range(0, N, batch):
        end = min(i+batch, N)
        payload = []
        for j in range(i, end):
            ch = chunks[j]
            meta = {
                "section": ch.get("section", ""),
                "page_start": int(ch.get("page_start", 0) or 0),
                "page_end": int(ch.get("page_end", 0) or 0),
                "text": ch.get("text", "")
            }
            payload.append((str(ch["chunk_id"]), vecs[j].tolist(), meta))
        if namespace:
            index.upsert(vectors=payload, namespace=namespace)
        else:
            index.upsert(vectors=payload)
