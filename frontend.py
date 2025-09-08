# frontend.py — Final premium dark/chat UI for AI Realtor Assistant
import streamlit as st
import requests
from requests.exceptions import RequestException

# ========== CONFIG ==========
# REPLACE this with your deployed backend URL (Render)
BACKEND_URL = " https://realestate-bot-backend.onrender.com"  # <-- set this
CHAT_ENDPOINT = f"{BACKEND_URL}/chat"
LEAD_ENDPOINT = f"{BACKEND_URL}/lead"

# ========== PAGE SETUP ==========
st.set_page_config(page_title="AI Realtor Assistant", page_icon="🏠", layout="wide", initial_sidebar_state="auto")

# ========== STYLES ==========
st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .header { text-align:center; margin-bottom: 6px; }
    .title { font-size:34px; color:#7dd3fc; text-shadow:0 0 12px #0891b2; margin:4px 0; }
    .subtitle { color:#94a3b8; margin-bottom:18px; }
    .user-bubble { background: linear-gradient(135deg,#1e3a8a,#3b82f6); color:white; padding:12px; border-radius:12px; margin:8px 0; text-align:right; box-shadow:0 0 10px #3b82f6; max-width:85%; margin-left:auto; }
    .bot-bubble  { background: linear-gradient(135deg,#0b1220,#1f2937); color:#e6eef6; padding:12px; border-radius:12px; margin:8px 0; text-align:left; box-shadow:0 0 12px #0ea5e9; max-width:85%; }
    .quick-btn { display:inline-block; background:#07143a; border:1px solid #2563eb; color:#c7f9ff; padding:8px 12px; border-radius:8px; margin:4px; cursor:pointer; }
    .debug { color:#94a3b8; font-size:12px; margin-top:6px; }
    @keyframes fadeIn { from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: translateY(0);} }
    .user-bubble, .bot-bubble { animation: fadeIn 0.35s ease-in-out; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========== HEADER ==========
st.markdown("<div class='header'><div class='title'>🤖 AI Realtor Assistant</div><div class='subtitle'>24/7 lead capture · tour booking · instant nurture</div></div>", unsafe_allow_html=True)

# ========== SESSION STATE ==========
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "awaiting_lead" not in st.session_state:
    st.session_state["awaiting_lead"] = False
if "lead_count" not in st.session_state:
    st.session_state["lead_count"] = 0
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None

# ========== LAYOUT ==========
chat_col, side_col = st.columns([4, 1])

# Sidebar content
with side_col:
    st.markdown("### ⚙ Bot Dashboard")
    backend_status = "Not configured" if "REPLACE_WITH_YOUR_RENDER_URL" in BACKEND_URL else "Configured"
    st.markdown(f"**Backend:** {backend_status}")
    st.markdown(f"**Leads captured:** {st.session_state['lead_count']}")
    st.markdown("---")
    st.markdown("### 💡 Quick asks")
    if st.button("Show me 2BHK under $500k"):
        q = "Show me 2BHK apartments under $500k"
        st.session_state["messages"].append({"role":"user","content":q})
        # call backend immediately
        try:
            r = requests.post(CHAT_ENDPOINT, json={"message": q}, timeout=10)
            bot = r.json().get("response", "No response")
        except RequestException as e:
            bot = f"Error contacting backend: {e}"
            st.session_state["last_error"] = str(e)
        st.session_state["messages"].append({"role":"assistant","content":bot})
        st.rerun()
    if st.button("Book a tour tomorrow 5pm"):
        q = "Book a tour for tomorrow at 5PM"
        st.session_state["messages"].append({"role":"user","content":q})
        try:
            r = requests.post(CHAT_ENDPOINT, json={"message": q}, timeout=10)
            bot = r.json().get("response", "No response")
        except RequestException as e:
            bot = f"Error contacting backend: {e}"
            st.session_state["last_error"] = str(e)
        st.session_state["messages"].append({"role":"assistant","content":bot})
        st.rerun()

    st.markdown("---")
    st.markdown("### ❓ What to ask")
    st.markdown("- Show me 2BHK apartments under $500k")
    st.markdown("- Book me a tour tomorrow at 5PM")
    st.markdown("- What’s the home buying process?")

# Chat column: show history
with chat_col:
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    # show any last error
    if st.session_state["last_error"]:
        st.markdown(f"<div class='debug'>Last error: {st.session_state['last_error']}</div>", unsafe_allow_html=True)

    # input box
    prompt = st.chat_input("Type your message...")

    if prompt:
        # register user message
        st.session_state["messages"].append({"role":"user","content":prompt})

        # if we are awaiting lead details, parse and send to /lead
        if st.session_state["awaiting_lead"]:
            parts = [p.strip() for p in prompt.split(",")]
            try:
                if len(parts) >= 3:
                    name = parts[0]; email = parts[1]; phone = parts[2]
                    budget = parts[3] if len(parts) >= 4 else "N/A"
                    payload = {"name": name, "email": email, "phone": phone, "budget": budget}
                    # debug visible to you
                    st.experimental_write("📤 Sending lead payload:", payload) if False else None
                    try:
                        resp = requests.post(LEAD_ENDPOINT, json=payload, timeout=10)
                        if resp.status_code == 200:
                            bot_reply = "✅ Thanks — I’ve saved your info. An agent will contact you soon."
                            st.session_state["lead_count"] += 1
                            st.session_state["awaiting_lead"] = False
                        else:
                            bot_reply = f"⚠️ Could not save lead (status {resp.status_code}): {resp.text}"
                            st.session_state["last_error"] = bot_reply
                    except RequestException as e:
                        bot_reply = f"Error contacting lead endpoint: {e}"
                        st.session_state["last_error"] = str(e)
                else:
                    bot_reply = "Please provide: Name, Email, Phone, Budget(optional). Example: John Doe, john@mail.com, +123456789, 500000"
            except Exception as e:
                bot_reply = f"Invalid input format: {e}"
        else:
            # normal chat -> send to /chat
            try:
                resp = requests.post(CHAT_ENDPOINT, json={"message": prompt}, timeout=12)
                if resp.status_code == 200:
                    bot_reply = resp.json().get("response", "No response from backend.")
                else:
                    bot_reply = f"Backend error (status {resp.status_code}): {resp.text}"
                    st.session_state["last_error"] = bot_reply
            except RequestException as e:
                bot_reply = f"Error contacting backend: {e}"
                st.session_state["last_error"] = str(e)

            # If bot suggests to save contact, flip awaiting_lead
            try:
                if isinstance(bot_reply, str) and ("save your contact" in bot_reply.lower() or "can i get your" in bot_reply.lower() or "share your contact" in bot_reply.lower()):
                    bot_reply += "\n\n💡 Please provide: Name, Email, Phone, Budget(optional)"
                    st.session_state["awaiting_lead"] = True
            except Exception:
                pass

        # append bot reply to history and rerun so UI updates instantly
        st.session_state["messages"].append({"role":"assistant","content":bot_reply})
        st.rerun()
