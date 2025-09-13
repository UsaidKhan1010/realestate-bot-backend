# app.py — RAG-enabled Real Estate Assistant (complete)
import os
import json
import re
from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv

# OpenAI client
from openai import OpenAI

# Google Sheets
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Local embeddings index utilities
import numpy as np

load_dotenv()

# -------------------------
# Config
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
BUILD_EMBEDDINGS_ON_START = os.getenv("BUILD_EMBEDDINGS_ON_START", "true").lower() in ("1", "true", "yes")

# Files
LISTINGS_CSV = "listings.csv"
EMB_FILE = "embeddings.npy"
META_FILE = "embeddings_meta.json"
FAQ_FILE = "faq.txt"  # optional extra grounding docs

# -------------------------
# OpenAI client init
# -------------------------
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("⚠️ OPENAI_API_KEY not set. LLM features will be disabled.")

# -------------------------
# Google Sheets init
# -------------------------
GOOGLE_CREDS_ENV = os.getenv("GOOGLE_CREDS")
SHEET_ID = os.getenv("SHEET_ID") or os.getenv("GOOGLE_SHEET_ID")
sheets_service = None
_sheets_ready = False
try:
    if GOOGLE_CREDS_ENV:
        creds_info = json.loads(GOOGLE_CREDS_ENV)
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        sheets_service = build("sheets", "v4", credentials=creds)
        _sheets_ready = True
        print("✅ Google Sheets credentials loaded from GOOGLE_CREDS env var.")
    elif os.path.exists("service_account.json"):
        creds = service_account.Credentials.from_service_account_file(
            "service_account.json", scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        sheets_service = build("sheets", "v4", credentials=creds)
        _sheets_ready = True
        print("✅ Google Sheets credentials loaded from local service_account.json.")
    else:
        print("⚠️ No Google creds found: service_account.json missing and GOOGLE_CREDS not set. Sheets will be disabled.")
except Exception as e:
    print("❌ Error initializing Google Sheets client:", e)
    _sheets_ready = False
    sheets_service = None

# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Data loading
# -------------------------
if not os.path.exists(LISTINGS_CSV):
    print("⚠️ listings.csv not found — creating sample listings.csv")
    sample = pd.DataFrame([
        {"id":1,"location":"Downtown","price":450000,"type":"Apartment","bedrooms":2,"description":"Great view","link":"http://example.com/1"},
        {"id":2,"location":"Midtown","price":520000,"type":"Apartment","bedrooms":3,"description":"Spacious","link":"http://example.com/2"},
        {"id":3,"location":"Suburbs","price":300000,"type":"House","bedrooms":2,"description":"Quiet area","link":"http://example.com/3"},
    ])
    sample.to_csv(LISTINGS_CSV, index=False)

listings = pd.read_csv(LISTINGS_CSV)

# -------------------------
# Simple in-memory leads (for quick debug)
# -------------------------
leads = []

# -------------------------
# Embeddings index utilities
# -------------------------
def build_embeddings_index(client: OpenAI, listings_csv: str = LISTINGS_CSV, faq_file: Optional[str] = FAQ_FILE):
    """
    Build embeddings (numpy array) and metadata JSON.
    """
    meta = []
    texts = []

    df = pd.read_csv(listings_csv)
    for i, row in df.iterrows():
        text = f"{row.get('location','')} | {row.get('bedrooms','')} BR | ${row.get('price','')} | {row.get('description','')}"
        meta.append({
            "id": f"listing_{i}",
            "source": "listing",
            "text": text,
            "location": row.get('location',''),
            "price": row.get('price',''),
            "bedrooms": int(row.get('bedrooms', 0)),
            "link": row.get('link', '')
        })
        texts.append(text)

    # optional extra FAQ blocks
    if faq_file and os.path.exists(faq_file):
        with open(faq_file, "r", encoding="utf-8") as f:
            blocks = [b.strip() for b in f.read().split("\n\n") if b.strip()]
        for i, block in enumerate(blocks):
            meta.append({"id": f"faq_{i}", "source": "faq", "text": block})
            texts.append(block)

    # compute embeddings in batches
    if client is None:
        raise RuntimeError("OpenAI client not configured for building embeddings.")
    vectors = []
    BATCH = 16
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        resp = client.embeddings.create(model=EMB_MODEL, input=batch)
        for d in resp.data:
            vectors.append(np.array(d.embedding, dtype=np.float32))

    arr = np.vstack(vectors)
    np.save(EMB_FILE, arr)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return len(meta)

def load_embeddings_index():
    if not os.path.exists(EMB_FILE) or not os.path.exists(META_FILE):
        return None, None
    emb = np.load(EMB_FILE)
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return emb, meta

def semantic_search(client: OpenAI, query: str, top_k: int = 4, emb=None, meta=None):
    """
    Return top_k meta items sorted by cosine similarity.
    """
    if client is None:
        return []
    if emb is None or meta is None:
        emb, meta = load_embeddings_index()
        if emb is None:
            return []

    resp = client.embeddings.create(model=EMB_MODEL, input=[query])
    qvec = np.array(resp.data[0].embedding, dtype=np.float32)

    # cosine similarity
    norm_emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    qnorm = qvec / np.linalg.norm(qvec)
    sims = (norm_emb @ qnorm).astype(np.float32)
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        results.append({"score": float(sims[i]), "meta": meta[i]})
    return results

# -------------------------
# Load or build embeddings on startup
# -------------------------
EMB_VEC, EMB_META = None, None

@app.on_event("startup")
def startup_event():
    global EMB_VEC, EMB_META
    EMB_VEC, EMB_META = load_embeddings_index()
    if EMB_VEC is None and client is not None and BUILD_EMBEDDINGS_ON_START:
        try:
            count = build_embeddings_index(client, listings_csv=LISTINGS_CSV, faq_file=FAQ_FILE)
            EMB_VEC, EMB_META = load_embeddings_index()
            print(f"Built embeddings index with {count} items.")
        except Exception as e:
            print("⚠️ Failed to build embeddings index on startup:", e)
    else:
        print("Embeddings index loaded." if EMB_VEC is not None else "No embeddings index available.")

# -------------------------
# Helper: save to Google Sheets
# -------------------------
def save_lead_to_sheet(lead: dict):
    if not _sheets_ready or sheets_service is None:
        raise RuntimeError("Google Sheets not configured on this server.")
    values = [[lead.get("name",""), lead.get("email",""), lead.get("phone",""), lead.get("budget","N/A")]]
    body = {"values": values}
    res = sheets_service.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range="Sheet1!A:D",
        valueInputOption="RAW",
        body=body
    ).execute()
    return res

# -------------------------
# System prompt / persona
# -------------------------
SYSTEM_PROMPT = (
    "You are Aiden — a professional, friendly, slightly confident real estate assistant who speaks like a helpful human agent.\n"
    "Rules:\n"
    "- Keep responses concise but useful (2–6 sentences) unless the user asks for a longer explainer.\n"
    "- Use natural language, contractions where natural.\n"
    "- Ask 1 relevant follow-up question when helpful (e.g. 'Which neighborhood are you interested in?').\n"
    "- When appropriate, end with a persuasive but honest CTA offering to save contact details or book a demo.\n"
    "- If the user asks for a property search, return up to 3 options when available in this format: location | price | short highlight | link.\n"
    "- When the user gives contact info, confirm politely and return a JSON object under key 'lead_capture' with keys: name, email, phone, budget.\n"
    "- Be human; do not say 'As an AI'."
)

# -------------------------
# Request models
# -------------------------
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = None  # optional conversation history [{"role":..,"content":..}]

class LeadRequest(BaseModel):
    name: str
    email: str
    phone: str
    budget: str = "N/A"

# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "Hello, bot is alive!"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_message = req.message.strip()
    history = req.history or []

    # Quick deterministic listing search for explicit 2BHK queries (fast response)
    if re.search(r"\b2bhk\b|\b2 bedroom\b|\b2-bedroom\b|\b2 bed\b", user_message, re.IGNORECASE):
        df = listings
        filtered = df[df.get('bedrooms', 0) == 2].head(3)
        if filtered.empty:
            return {"response": "I couldn't find 2BHK listings right now — which area should I search?"}
        response_text = "Here are a few 2BHK options you might like:\n"
        for _, row in filtered.iterrows():
            highlight = row.get("description", "") or "Great spot"
            response_text += f"- {row.get('location','')} | ${int(row.get('price',0)):,} | {highlight} | {row.get('link','')}\n"
        response_text += "\nWant me to save your contact so an agent can reach out?"
        return {"response": response_text}

    # RAG: semantic search for grounding context
    contexts = []
    try:
        results = semantic_search(client, user_message, top_k=4, emb=EMB_VEC, meta=EMB_META)
        for r in results:
            m = r["meta"]
            contexts.append(f"[{m.get('source','')}] {m.get('text','')}")
    except Exception:
        contexts = []

    # Compose system prompt + contexts
    system_with_context = SYSTEM_PROMPT
    if contexts:
        system_with_context += "\n\nUse these facts to ground your answer (do not invent facts):\n"
        for c in contexts:
            system_with_context += "- " + c.replace("\n", " ").strip()[:800] + "\n"

    # Build messages: include recent history if provided
    messages = [{"role":"system","content":system_with_context}]
    if history:
        for m in history[-6:]:
            messages.append(m)
    messages.append({"role":"user","content":user_message})

    # Call LLM
    if client is None:
        return {"response": "Sorry — the AI engine is not configured (OPENAI_API_KEY missing)."}
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=600
        )
        assistant_text = completion.choices[0].message.content.strip()
    except Exception as e:
        return {"response": f"GPT error: {str(e)}"}

    # Post-process to detect lead_capture JSON if model returns it
    lead_capture = None
    try:
        jmatch = re.search(r'(\{[^}]*"lead_capture"[^}]*\})', assistant_text, re.DOTALL)
        if jmatch:
            data = json.loads(jmatch.group(1))
            if "lead_capture" in data:
                lead_capture = data["lead_capture"]
    except Exception:
        lead_capture = None

    # Also detect raw contact lines from user's last message (e.g., "John, john@example.com, +123..., 500000")
    if not lead_capture:
        parts = [p.strip() for p in re.split(r'[,\n;|]+', user_message) if p.strip()]
        if len(parts) >= 3 and re.search(r"@", parts[1]) and re.search(r"\+?\d", parts[2]):
            lead_capture = {
                "name": parts[0],
                "email": parts[1],
                "phone": parts[2],
                "budget": parts[3] if len(parts) > 3 else "N/A"
            }
            assistant_text = f"Got it — I’ll save {lead_capture['name']}, {lead_capture['email']}, {lead_capture['phone']} (budget: {lead_capture['budget']}). An agent will reach out soon."

    resp = {"response": assistant_text}
    if lead_capture:
        resp["lead_capture"] = lead_capture

    return resp

@app.post("/lead")
async def save_lead(lead: LeadRequest):
    lead_dict = lead.dict()
    leads.append(lead_dict)
    print("📌 /lead received:", lead_dict)

    if _sheets_ready:
        try:
            res = save_lead_to_sheet(lead_dict)
            print("✅ Lead saved to Google Sheets. Response:", res.get("updates", "ok") if isinstance(res, dict) else res)
        except Exception as e:
            print("❌ Sheets error:", e)
            return {"message": f"Lead saved locally but Sheets error: {str(e)}", "total_leads": len(leads), "latest_lead": lead_dict}
    else:
        print("⚠️ Sheets not configured — lead only saved locally.")

    return {
        "message": "Lead saved successfully (and pushed to Google Sheets)" if _sheets_ready else "Lead saved locally (Sheets not configured)",
        "total_leads": len(leads),
        "latest_lead": lead_dict
    }

# -------------------------
# Utility endpoints (optional)
# -------------------------
@app.post("/rebuild-index")
def rebuild_index_endpoint():
    """
    Trigger rebuilding embeddings index on demand (admin use).
    Requires OPENAI_API_KEY and will call embeddings API (cost).
    """
    if client is None:
        return {"ok": False, "error": "OpenAI not configured"}
    try:
        count = build_embeddings_index(client, listings_csv=LISTINGS_CSV, faq_file=FAQ_FILE if os.path.exists(FAQ_FILE) else None)
        return {"ok": True, "count": count}
    except Exception as e:
        return {"ok": False, "error": str(e)}

