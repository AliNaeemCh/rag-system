import streamlit as st



def render_hints():

    hint_slot = st.empty()



    if (
        not st.session_state.chat_started
        and not st.session_state.get(
            "pending_input"
        )
    ):


        with hint_slot.container():

            st.markdown(
                "### Try asking:"
            )


            hints = [

                "What is the core business of Systems Ltd.?",
                "Who is the CEO of Systems Ltd.?",
                "What was Systems Ltd.’s profit in 2025?"

            ]



            for q in hints:


                if st.button(
                    q,
                    key=q
                ):


                    st.session_state.chat_started = True

                    st.session_state.pending_input = q

                    st.rerun()


    else:

        hint_slot.empty()