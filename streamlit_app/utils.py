"""Helper functions for the Streamlit chat UI. No LLM/business logic here —
that all lives in app.modules and is invoked identically from app/main.py."""
from datetime import datetime, timedelta

import streamlit as st

OPENING_MESSAGE = (
    "Thanks for applying to our Python Developer opening. "
    "What kinds of Python projects have you worked on recently?"
)


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = [{"speaker": "recruiter", "text": OPENING_MESSAGE}]
    if "simulated_now" not in st.session_state:
        st.session_state.simulated_now = datetime.utcnow()
    if "ended" not in st.session_state:
        st.session_state.ended = False


def reset_conversation() -> None:
    st.session_state.history = [{"speaker": "recruiter", "text": OPENING_MESSAGE}]
    st.session_state.simulated_now = datetime.utcnow()
    st.session_state.ended = False


def advance_simulated_clock(minutes: int = 30) -> None:
    st.session_state.simulated_now += timedelta(minutes=minutes)


def render_message(speaker: str, text: str) -> None:
    role = "assistant" if speaker == "recruiter" else "user"
    with st.chat_message(role):
        st.write(text)
