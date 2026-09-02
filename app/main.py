"""CLI entry point for the SMS job-candidate chatbot, and the shared agent
wiring helper reused by streamlit_app/streamlit_main.py and the eval notebook."""
from datetime import datetime, timezone

from app import config
from app.modules.exit_advisor.m_2 import ConversationExitAdvisor
from app.modules.info_advisor.m_4 import ConversationInfoAdvisor
from app.modules.main_agent.m_1 import MainAgent
from app.modules.scheduling_advisor.m_3 import InterviewSchedulingAdvisor
from app.modules.scheduling_db.repository import ScheduleRepository


def build_main_agent(db_path: str | None = None, chroma_dir: str | None = None) -> MainAgent:
    repository = ScheduleRepository(db_path or config.SCHEDULE_DB_PATH)
    return MainAgent(
        exit_advisor=ConversationExitAdvisor(),
        scheduling_advisor=InterviewSchedulingAdvisor(repository=repository, position=config.DEFAULT_POSITION),
        info_advisor=ConversationInfoAdvisor(chroma_dir=chroma_dir or config.CHROMA_DIR),
        repository=repository,
    )


def run_cli() -> None:
    agent = build_main_agent()
    history: list[dict] = []

    print("Recruiter bot ready. Type a candidate message (or 'quit' to exit).\n")
    opening = (
        "Thanks for applying to our Python Developer opening. "
        "What kinds of Python projects have you worked on recently?"
    )
    print(f"Recruiter: {opening}")
    history.append({"speaker": "recruiter", "text": opening})

    while True:
        candidate_message = input("Candidate: ").strip()
        if candidate_message.lower() in {"quit", "exit"}:
            break

        history.append({"speaker": "candidate", "text": candidate_message})
        conversation_now = datetime.now(timezone.utc).replace(tzinfo=None)
        decision = agent.step(history, candidate_message, conversation_now)
        print(f"Recruiter: {decision.reply_text}")
        history.append({"speaker": "recruiter", "text": decision.reply_text})

        if decision.action == "end":
            break


if __name__ == "__main__":
    run_cli()
