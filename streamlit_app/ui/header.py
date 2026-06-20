import streamlit as st

from app.core.config import settings


def render_header():

    st.markdown(
        f"""
        <h1 style="margin-bottom:5px;">
            {settings.TITLE}
        </h1>

        <p style="color:gray;font-size:16px;">
            {settings.SUB_TITLE}
        </p>
        """,
        unsafe_allow_html=True
    )