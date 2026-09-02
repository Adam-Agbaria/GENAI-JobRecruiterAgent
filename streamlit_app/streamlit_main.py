"""Streamlit PoC UI for the SMS job-candidate chatbot (deploy target: Streamlit
Community Cloud, standing in for real SMS integration). Pure UI glue over
app.main.build_main_agent() — no LLM calls or business logic live here."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.main import build_main_agent
from streamlit_app.utils import init_session_state, render_message, reset_conversation

st.set_page_config(page_title="Python Developer Recruiting Bot", page_icon="\U0001F4AC")


@st.cache_resource
def get_agent():
    return build_main_agent()


init_session_state()
agent = get_agent()

st.title("Python Developer Recruiting Bot")
st.caption("Proof-of-concept chat UI standing in for SMS. Play the role of the candidate.")

if st.button("Reset conversation"):
    reset_conversation()
    st.rerun()

for turn in st.session_state.history:
    render_message(turn["speaker"], turn["text"])

if st.session_state.ended:
    st.info("Conversation has ended.")
else:
    candidate_message = st.chat_input("Type the candidate's reply...")
    if candidate_message:
        st.session_state.history.append({"speaker": "candidate", "text": candidate_message})
        render_message("candidate", candidate_message)

        conversation_now = datetime.now(timezone.utc).replace(tzinfo=None)
        decision = agent.step(st.session_state.history, candidate_message, conversation_now)
        st.session_state.history.append({"speaker": "recruiter", "text": decision.reply_text})
        render_message("recruiter", decision.reply_text)

        if decision.action == "end":
            st.session_state.ended = True
