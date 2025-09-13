# build_embeddings_local.py
import os, json
import numpy as np
import pandas as pd
from openai import OpenAI

EMB_MODEL = "text-embedding-3-small"
LISTINGS_CSV = "listings.csv"
EMB_FILE = "embeddings.npy"
META_FILE = "embeddings_meta.json"

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in your environment before running this (instructions below).")
    client = OpenAI(api_key=api_key)

    # load listings
    if not os.path.exists(LISTINGS_CSV):
        raise SystemExit(f"{LISTINGS_CSV} not found in current directory.")
    df = pd.read_csv(LISTINGS_CSV)

    texts = []
    meta = []
    for i, row in df.iterrows():
        text = f"{row.get('location','')} | {row.get('bedrooms','')} BR | ${row.get('price','')} | {row.get('description','')}"
        meta.append({
            "id": f"listing_{i}",
            "source": "listing",
            "text": text,
            "location": row.get("location",""),
            "price": row.get("price",""),
            "bedrooms": int(row.get("bedrooms",0)),
            "link": row.get("link","")
        })
        texts.append(text)

    # batch call
    embeddings = []
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMB_MODEL, input=batch)
        for d in resp.data:
            embeddings.append(np.array(d.embedding, dtype=np.float32))

    arr = np.vstack(embeddings)
    np.save(EMB_FILE, arr)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("✅ Wrote", EMB_FILE, "and", META_FILE)

if __name__ == "__main__":
    main()
