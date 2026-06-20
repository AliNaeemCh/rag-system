from app.core.config import settings
from streamlit_app.services.chat_api import stream_chat
from streamlit_app.constants.typing import TYPING_HTML

import time
import streamlit as st

def init_session():


    if "session_id" not in st.session_state:

        st.session_state.session_id = None



    if "messages" not in st.session_state:

        st.session_state.messages = []



    if "chat_started" not in st.session_state:

        st.session_state.chat_started = False

def render_history():


    for msg in st.session_state.messages:


        with st.chat_message(
            msg["role"]
        ):

            st.write(
                msg["content"]
            )

def render_chat(mode):


    init_session()


    render_history()



    pending_input = st.session_state.pop(
        "pending_input",
        None
    )


    typed_input = st.chat_input(
        "Ask something..."
    )



    user_input = (
        pending_input
        or typed_input
    )



    if not user_input:

        return



    st.session_state.chat_started = True



    if len(user_input) > settings.USER_IN_MAX_CHARS:

        st.error(
            f"❌ Input must be {settings.USER_IN_MAX_CHARS} characters or less."
        )

        st.stop()



    st.session_state.messages.append(

        {
            "role":"user",
            "content":user_input
        }

    )



    with st.chat_message("user"):

        st.write(user_input)




    assistant_box = st.chat_message(
        "assistant"
    )


    placeholder = assistant_box.empty()



    placeholder.markdown(
        TYPING_HTML,
        unsafe_allow_html=True
    )



    full_response = ""



    try:


        for event in stream_chat(

            user_input,
            st.session_state.session_id,
            mode

        ):


            event_type = event.get(
                "type"
            )



            if event_type == "session":


                st.session_state.session_id = (
                    event.get("session_id")
                )



            elif event_type == "token":


                full_response += event.get(
                    "token",
                    ""
                )


                placeholder.markdown(
                    full_response + "▌"
                )


                time.sleep(.04)



            elif event_type == "done":

                break



            elif event_type == "error":

                raise Exception(
                    f"Error from backend. Event: {event}"
                )




        placeholder.markdown(
            full_response
        )



    except Exception as e:


        print(e)


        full_response = (
            "⚠️ Sorry, I couldn’t generate a response right now. "
            "Please try again."
        )


        placeholder.markdown(
            full_response
        )




    st.session_state.messages.append(

        {
            "role":"assistant",
            "content":full_response
        }

    )