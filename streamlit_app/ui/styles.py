import streamlit as st



def load_styles():

    st.markdown("""
    <style>

    html, body {
        font-size: 18px !important;
    }


    [data-testid="stChatMessage"] * {
        font-size: 20px !important;
        line-height: 1.6 !important;
    }

    </style>
    """,
    unsafe_allow_html=True)



    st.markdown("""
    <style>

    [data-testid="stSidebar"] {

        width:210px !important;
        min-width:210px !important;

    }


    [data-testid="stSidebar"] > div:first-child {

        padding-top:.8rem;
        padding-left:.6rem;
        padding-right:.6rem;

    }


    [data-testid="stRadio"] label {

        font-size:13px !important;

    }

    </style>
    """,
    unsafe_allow_html=True)