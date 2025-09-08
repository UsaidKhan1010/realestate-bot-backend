import streamlit as st
import requests

st.set_page_config(page_title="AI Realtor Assistant", page_icon="🏠", layout="wide")

# ==========================
# Custom Dark Mode CSS
# ==========================
st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .user-bubble {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: right;
        box-shadow: 0px 0px 10px #3b82f6;
        animation: fadeIn 0.5s ease-in-out;
    }
    .bot-bubble {
        background: linear-gradient(135deg, #1e293b, #334155);
        color: #f8fafc;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        text-align: left;
        box-shadow: 0px 0px 10px #0ea5e9;
        animation: fadeIn 0.5s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align:center; color:#38bdf8;'>🤖 AI Realtor Assistant</h1>", unsafe_allow_html=True)

# ==========================
# Session State
# ==========================
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "awaiting_lead" not in st.session_state:
    st.session_state["awaiting_lead"] = False

# ==========================
# Show Chat History
# ==========================
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

# ==========================
# Chat Input
# ==========================
if prompt := st.chat_input("Type your message..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    if st.session_state["awaiting_lead"]:
        try:
            parts = [p.strip() for p in prompt.split(",")]
            if len(parts) == 4:
                name, email, phone, budget = parts
            elif len(parts) == 3:
                name, email, phone = parts
                budget = "N/A"
            else:
                raise ValueError("Invalid input format")

            payload = {"name": name, "email": email, "phone": phone, "budget": budget}
            st.write("📤 Debug payload:", payload)  # Debug log in UI

            response = requests.post("http://127.0.0.1:8000/lead", json=payload)

            if response.status_code == 200:
                bot_reply = "✅ Thanks! I’ve saved your info. An agent will reach out soon."
                st.session_state["awaiting_lead"] = False
            else:
                bot_reply = "⚠️ Couldn't save your info. Please try again."
        except Exception as e:
            bot_reply = f"⚠️ Invalid input: {e}. Please type **Name, Email, Phone, Budget(optional)**"
    else:
        try:
            response = requests.post("http://127.0.0.1:8000/chat", json={"message": prompt})
            bot_reply = response.json().get("response", "⚠️ No response")

            if "save your contact" in bot_reply.lower():
                bot_reply += "\n\n💡 Can I get your **Name, Email, Phone, Budget(optional)**?"
                st.session_state["awaiting_lead"] = True

        except Exception as e:
            bot_reply = f"Error contacting backend: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
    st.rerun()
