import streamlit as st



def render_sidebar():

    with st.sidebar:

        st.markdown(
            "### ⚙️ Response Mode"
        )


        mode = st.radio(
            "Response Mode",
            [
                "⚡ Fast",
                "⚖️ Balanced",
                "🧠 Advanced"
            ],
            index=1,
            label_visibility="collapsed"
        )


    return mode.split(" ",1)[1].lower()