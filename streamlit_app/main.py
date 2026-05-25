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

st.set_page_config(page_title=settings.TITLE, layout="centered")
st.title(settings.TITLE)

# ----------------------------
# session state
# ----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# render chat history
# ----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ----------------------------
# input
# ----------------------------
user_input = st.chat_input("Ask something...")

# ----------------------------
# CSS typing animation (WhatsApp style)
# ----------------------------
TYPING_HTML = """
<div style="display:flex; gap:5px; align-items:center;">
  <div class="dot"></div>
  <div class="dot"></div>
  <div class="dot"></div>
</div>

<style>
.dot {
  width: 7px;
  height: 7px;
  background-color: #999;
  border-radius: 50%;
  display: inline-block;
  animation: bounce 1.2s infinite ease-in-out;
}

.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
  40% { transform: scale(1); opacity: 1; }
}
</style>
"""

if user_input:

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
        "session_id": st.session_state.session_id
    }

    try:
        with requests.post(
            settings.CHAT_URL,
            json=payload,
            stream=True,
            timeout=300
        ) as res:

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

        # final render (remove cursor)
        placeholder.markdown(full_response)

    except Exception as e:
        placeholder.markdown(f"Error: {str(e)}")

    # save history
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )