# app.py
import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from dotenv import load_dotenv

# OpenAI new client
from openai import OpenAI

# Google Sheets
from google.oauth2 import service_account
from googleapiclient.discovery import build

# --------------------------
# Load .env (optional)
# --------------------------
load_dotenv()

# --------------------------
# OpenAI client setup
# --------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not set. GPT fallback will fail if used.")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# --------------------------
# Google Sheets setup (dual: env or local file)
# --------------------------
# Priority:
# 1) If GOOGLE_CREDS env var exists -> use it (for Render)
# 2) Else if service_account.json file exists locally -> use it (for local dev)
GOOGLE_CREDS_ENV = os.getenv("GOOGLE_CREDS")  # full JSON string (Render)
SHEET_ID = os.getenv("SHEET_ID") or os.getenv("GOOGLE_SHEET_ID") or "1-54BdoVXiH0_pTrDSy30HgW16ob6o3Rs4RUiZvELkPI"

service = None
_sheets_ready = False

try:
    if GOOGLE_CREDS_ENV:
        # load from env JSON
        creds_info = json.loads(GOOGLE_CREDS_ENV)
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)
        _sheets_ready = True
        print("✅ Google Sheets credentials loaded from GOOGLE_CREDS env var.")
    elif os.path.exists("service_account.json"):
        # load from local file (dev)
        creds = service_account.Credentials.from_service_account_file(
            "service_account.json", scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        service = build("sheets", "v4", credentials=creds)
        _sheets_ready = True
        print("✅ Google Sheets credentials loaded from local service_account.json.")
    else:
        print("⚠️ No Google creds found: service_account.json missing and GOOGLE_CREDS not set. Sheets will be disabled.")
except Exception as e:
    print("❌ Error initializing Google Sheets client:", str(e))
    _sheets_ready = False
    service = None

# --------------------------
# FastAPI init + CORS
# --------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for demo. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# Request models
# --------------------------
class ChatRequest(BaseModel):
    message: str

class LeadRequest(BaseModel):
    name: str
    email: str
    phone: str
    budget: str = "N/A"

# --------------------------
# Data loading
# --------------------------
# ensure listings.csv exists. If not, create a small sample in memory.
LISTINGS_CSV = "listings.csv"
if not os.path.exists(LISTINGS_CSV):
    print("⚠️ listings.csv not found — creating sample listings.csv")
    sample = pd.DataFrame([
        {"id":1,"location":"Downtown","price":450000,"type":"Apartment","bedrooms":2,"link":"http://example.com/1"},
        {"id":2,"location":"Midtown","price":520000,"type":"Apartment","bedrooms":3,"link":"http://example.com/2"},
        {"id":3,"location":"Suburbs","price":300000,"type":"House","bedrooms":2,"link":"http://example.com/3"},
    ])
    sample.to_csv(LISTINGS_CSV, index=False)

listings = pd.read_csv(LISTINGS_CSV)
leads = []  # in-memory store for quick debugging

# --------------------------
# Helper: save lead to Google Sheets (if available)
# --------------------------
def save_lead_to_sheet(lead: dict):
    if not _sheets_ready or service is None:
        raise RuntimeError("Google Sheets not configured on this server.")
    values = [[
        lead.get("name", ""),
        lead.get("email", ""),
        lead.get("phone", ""),
        lead.get("budget", "N/A")
    ]]
    body = {"values": values}
    print("📤 Sending to Sheets:", values)
    result = service.spreadsheets().values().append(
        spreadsheetId=SHEET_ID,
        range="Sheet1!A:D",
        valueInputOption="RAW",
        body=body
    ).execute()
    return result

# --------------------------
# Routes
# --------------------------
@app.get("/")
def read_root():
    return {"message": "Hello, bot is alive!"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    print("💬 /chat received:", user_message)

    # Simple listings rule
    lower = user_message.lower()
    if "2bhk" in lower or "2 bedroom" in lower or "2-bed" in lower:
        filtered = listings[listings.get('bedrooms', 0) == 2].head(3)
        if filtered.empty:
            return {"response": "I looked but couldn't find 2BHK options right now. Want me to alert you when one appears?"}
        response_text = "Here are a few 2BHK options you might like:\n"
        for _, row in filtered.iterrows():
            response_text += f"- {row['location']} | ${row['price']} | {row['link']}\n"
        response_text += "\nWould you like me to save your contact details so an agent can share more?"
        return {"response": response_text}

    # Otherwise use OpenAI if available
    if client is None:
        return {"response": "Sorry — the AI engine is not configured (OPENAI_API_KEY missing)."}
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a friendly real estate assistant. Speak naturally and conversationally."},
                {"role": "user", "content": user_message}
            ]
        )
        text = completion.choices[0].message.content
        print("🧠 GPT reply:", text)
        return {"response": text}
    except Exception as e:
        print("❌ GPT error:", str(e))
        return {"response": f"Oops — I had trouble answering that: {str(e)}"}

@app.post("/lead")
async def save_lead(lead: LeadRequest):
    lead_dict = lead.dict()
    leads.append(lead_dict)
    print("📌 /lead received:", lead_dict)

    if _sheets_ready:
        try:
            res = save_lead_to_sheet(lead_dict)
            print("✅ Lead saved to Google Sheets. API response:", getattr(res, "get", lambda k: None)("updates", "ok"))
        except Exception as e:
            print("❌ Sheets error:", e)
            return {"message": f"Lead saved locally but Sheets error: {str(e)}"}
    else:
        print("⚠️ Sheets not configured — lead only saved locally.")

    return {
        "message": "Lead saved successfully (and pushed to Google Sheets)" if _sheets_ready else "Lead saved locally (Sheets not configured)",
        "total_leads": len(leads),
        "latest_lead": lead_dict
    }

