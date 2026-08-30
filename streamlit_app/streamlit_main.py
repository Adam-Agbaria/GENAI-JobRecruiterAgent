"""Streamlit PoC UI for the SMS job-candidate chatbot (deploy target: Streamlit
Community Cloud, standing in for real SMS integration). Pure UI glue over
app.main.build_main_agent() — no LLM calls or business logic live here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.main import build_main_agent
from streamlit_app.utils import (
    advance_simulated_clock,
    init_session_state,
    render_message,
    reset_conversation,
)

st.set_page_config(page_title="Python Developer Recruiting Bot", page_icon="\U0001F4AC")


@st.cache_resource
def get_agent():
    return build_main_agent()


init_session_state()
agent = get_agent()

st.title("Python Developer Recruiting Bot")
st.caption("Proof-of-concept chat UI standing in for SMS. Play the role of the candidate.")

with st.sidebar:
    st.subheader("Simulated conversation clock")
    st.write(st.session_state.simulated_now.strftime("%Y-%m-%d %H:%M UTC"))
    st.caption("Advance the clock to test relative-date scheduling ('next Friday', etc.)")
    if st.button("Advance +1 day"):
        advance_simulated_clock(minutes=24 * 60)
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

        decision = agent.step(st.session_state.history, candidate_message, st.session_state.simulated_now)
        st.session_state.history.append({"speaker": "recruiter", "text": decision.reply_text})
        render_message("recruiter", decision.reply_text)

        advance_simulated_clock(minutes=30)
        if decision.action == "end":
            st.session_state.ended = True

        with st.expander("Debug: advisor decision trace"):
            st.json({"action": decision.action, **decision.meta})
