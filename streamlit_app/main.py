import sys
import json
from pathlib import Path
import requests
import streamlit as st
import time

# ----------------------------
# project setup
# ----------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.core.config import settings

st.markdown("""
<style>
/* Global font size */
html, body {
    font-size: 18px !important;
}

/* Chat message text */
[data-testid="stChatMessage"] * {
    font-size: 20px !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title=settings.TITLE, layout="centered")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    width: 210px !important;
    min-width: 210px !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 0.8rem;
    padding-left: 0.6rem;
    padding-right: 0.6rem;
}

[data-testid="stRadio"] label {
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="padding: 10px 0 20px 0;">
    <h1 style="margin-bottom: 5px;">{settings.TITLE}</h1>
    <div style="color: gray; font-size: 16px;">
        {settings.SUB_TITLE}
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Response Mode")

    mode = st.radio(
        "Response Mode",
        ["⚡ Fast", "⚖️ Balanced", "🧠 Advanced"],
        index=1,
        label_visibility="collapsed"
    )

    mode = mode.split(" ", 1)[1].lower()
    
# ----------------------------
# session state
# ----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# ----------------------------
# render chat history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ----------------------------
# input
# ----------------------------
pending_input = st.session_state.pop("pending_input", None)

# ALWAYS render chat_input
typed_input = st.chat_input("Ask something...")

# choose pending hint OR typed message
user_input = pending_input or typed_input

if user_input:
    st.session_state.chat_started = True

# ----------------------------
# hint questions
# ----------------------------
hint_slot = st.empty()

if not st.session_state.chat_started and not user_input:
    with hint_slot.container():
        st.markdown("### Try asking:")

        hints = [
            "What is the core business of Systems Ltd.?",
            "Who is the CEO of Systems Ltd.?",
            "What was Systems Ltd.’s profit in 2025?"
        ]

        for q in hints:
            if st.button(q, key=q):
                st.session_state.chat_started = True
                st.session_state.pending_input = q
                st.rerun()
else:
    hint_slot.empty()

# ----------------------------
# CSS typing animation (WhatsApp style)
# ----------------------------
TYPING_HTML = """<div class="typing">
  <span></span>
  <span></span>
  <span></span>
</div>

<style>
.typing {
  display: flex;
  align-items: center;
  gap: 6px;
}

.typing span {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #999;

  animation: typing 1.6s infinite ease-in-out;
}

.typing span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.45;
  }

  30% {
    transform: translateY(-4px);
    opacity: 1;
  }
}
</style>"""

if user_input:

    if len(user_input) > settings.USER_IN_MAX_CHARS:
        st.error(f"❌ Input must be {settings.USER_IN_MAX_CHARS} characters or less.")
        st.stop()   # stops execution for this run

    # store + show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    # assistant UI
    assistant_box = st.chat_message("assistant")
    placeholder = assistant_box.empty()

    # show smooth typing animation immediately (browser-side CSS)
    placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)

    full_response = ""

    payload = {
        "message": user_input,
        "session_id": st.session_state.session_id,
        "mode": mode
    }

    headers = {
        "X-API-Key": settings.API_KEY
    }

    try:
        with requests.post(
            settings.CHAT_URL,
            json=payload,
            headers=headers,
            stream=True,
            timeout=300
        ) as res:
            if res.status_code and res.status_code >= 400:
                raise
            for line in res.iter_lines(decode_unicode=True):

                if not line:
                    continue

                if line.startswith("data: "):
                    event = json.loads(line.replace("data: ", ""))
                    event_type = event.get("type")

                    # session
                    if event_type == "session":
                        st.session_state.session_id = event.get("session_id")

                    # streaming tokens
                    elif event_type == "token":
                        full_response += event.get("token", "")

                        # replace typing animation with text
                        placeholder.markdown(full_response + "▌")

                        time.sleep(0.04)

                    elif event_type == "done":
                        break

                    elif event_type == "error":
                        raise

        # final render (remove cursor)
        placeholder.markdown(full_response)

    except Exception:
        full_response = "⚠️ Sorry, I couldn’t generate a response right now. Please try again."
        placeholder.markdown(full_response)

    # save history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

