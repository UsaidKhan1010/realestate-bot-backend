# embeddings_index.py
import os, json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

EMB_FILE = "embeddings.npy"
META_FILE = "embeddings_meta.json"
EMB_MODEL = "text-embedding-3-small"  # OpenAI embedding model

def build_index(client: OpenAI, listings_csv="listings.csv", extra_text_file=None):
    """
    Build embeddings for each listing row + optional extra texts.
    Saves numpy embeddings and metadata JSON.
    """
    meta = []   # list of dicts: {id, source, text, extra fields}
    vectors = []

    # load listings
    df = pd.read_csv(listings_csv)
    for i, row in df.iterrows():
        text = f"{row.get('location','')} | {row.get('bedrooms','')} BR | ${row.get('price','')} | {row.get('description','')}"
        meta.append({
            "id": f"listing_{i}",
            "source": "listing",
            "text": text,
            "location": row.get('location',''),
            "price": row.get('price',''),
            "bedrooms": int(row.get('bedrooms',0)),
            "link": row.get('link','')
        })
        vectors.append(text)

    # load extra texts (FAQ, seller tips)
    if extra_text_file and os.path.exists(extra_text_file):
        with open(extra_text_file, "r", encoding="utf-8") as f:
            for i, block in enumerate(f.read().split("\n\n")):
                block = block.strip()
                if not block:
                    continue
                meta.append({
                    "id": f"faq_{i}",
                    "source": "faq",
                    "text": block
                })
                vectors.append(block)

    # call OpenAI embeddings in batches
    batched_embs = []
    BATCH = 16
    for i in range(0, len(vectors), BATCH):
        batch = vectors[i:i+BATCH]
        resp = client.embeddings.create(model=EMB_MODEL, input=batch)
        for d in resp.data:
            batched_embs.append(np.array(d.embedding, dtype=np.float32))

    arr = np.vstack(batched_embs)
    np.save(EMB_FILE, arr)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return len(meta)

def load_index():
    """
    Returns (embeddings ndarray, meta list)
    """
    if not os.path.exists(EMB_FILE) or not os.path.exists(META_FILE):
        return None, None
    emb = np.load(EMB_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return emb, meta

def semantic_search(client: OpenAI, query: str, top_k=5, emb=None, meta=None):
    """
    Given query string, return top_k metadata items with scores.
    """
    if emb is None or meta is None:
        emb, meta = load_index()
        if emb is None:
            return []

    resp = client.embeddings.create(model=EMB_MODEL, input=[query])
    qvec = np.array(resp.data[0].embedding, dtype=np.float32)

    # cosine similarity
    # emb shape: (N, dim)
    norm_emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    qnorm = qvec / np.linalg.norm(qvec)
    sims = (norm_emb @ qnorm).astype(np.float32)  # dot product = cos sim
    idx = np.argsort(-sims)[:top_k]

    results = []
    for i in idx:
        results.append({"score": float(sims[i]), "meta": meta[i]})
    return results
