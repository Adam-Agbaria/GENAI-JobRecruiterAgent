"""Shared conversation-history helpers used by every advisor, the Main Agent,
the CLI, the Streamlit app, and the evaluation notebook, so the notion of a
"turn" and how it's rendered into a prompt stays consistent everywhere."""
from dataclasses import dataclass


@dataclass
class Turn:
    speaker: str  # "recruiter" | "candidate"
    text: str
    timestamp_utc: str | None = None


def to_turn(turn: "Turn | dict") -> Turn:
    if isinstance(turn, Turn):
        return turn
    return Turn(speaker=turn["speaker"], text=turn["text"], timestamp_utc=turn.get("timestamp_utc"))


def format_history(turns: list) -> str:
    """Renders a list of turns as "Role: text" lines for use inside a prompt."""
    lines = []
    for raw in turns:
        t = to_turn(raw)
        role = "Recruiter" if t.speaker == "recruiter" else "Candidate"
        lines.append(f"{role}: {t.text}")
    return "\n".join(lines) if lines else "(no prior messages)"


def last_candidate_message(turns: list) -> str | None:
    for raw in reversed(turns):
        t = to_turn(raw)
        if t.speaker == "candidate":
            return t.text
    return None
