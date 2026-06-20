import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from app.core.config import settings

from streamlit_app.ui.styles import load_styles
from streamlit_app.ui.header import render_header
from streamlit_app.ui.sidebar import render_sidebar
from streamlit_app.ui.chat import render_chat
from streamlit_app.ui.hints import render_hints


st.set_page_config(
    page_title=settings.TITLE,
    layout="centered"
)


load_styles()

render_header()

mode = render_sidebar()

render_chat(mode)

render_hints()