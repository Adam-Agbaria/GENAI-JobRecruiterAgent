"""
Main Agent: orchestrates the conversation turn-by-turn.

Deterministic fusion of the three advisors (Exit > Scheduling > Info priority),
so the same `step()` call can be reused by the CLI, the Streamlit app, and the
evaluation notebook, and its decisions can be unit-tested with mocked advisors.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from app.conversation import format_history
from app.modules.exit_advisor.m_2 import ConversationExitAdvisor
from app.modules.info_advisor.m_4 import ConversationInfoAdvisor
from app.modules.scheduling_advisor.m_3 import InterviewSchedulingAdvisor, ProposedSlot
from app.modules.scheduling_db.repository import ScheduleRepository

Action = Literal["continue", "schedule", "end"]


@dataclass
class AgentDecision:
    action: Action
    reply_text: str
    meta: dict = field(default_factory=dict)


def _format_slot(slot: ProposedSlot) -> str:
    return f"{slot.date} at {slot.time[:5]}"


class MainAgent:
    def __init__(
        self,
        exit_advisor: ConversationExitAdvisor,
        scheduling_advisor: InterviewSchedulingAdvisor,
        info_advisor: ConversationInfoAdvisor,
        repository: ScheduleRepository,
        exit_confidence_threshold: float = 0.5,
    ):
        self.exit_advisor = exit_advisor
        self.scheduling_advisor = scheduling_advisor
        self.info_advisor = info_advisor
        self.repository = repository
        self.exit_confidence_threshold = exit_confidence_threshold

    def step(self, conversation_history: list, candidate_message: str | None, conversation_now: datetime) -> AgentDecision:
        history_text = format_history(conversation_history)
        candidate_message = candidate_message or "(conversation just started)"

        exit_decision = self.exit_advisor.decide(history_text, candidate_message)
        if exit_decision.should_end and exit_decision.confidence >= self.exit_confidence_threshold:
            return AgentDecision(
                action="end",
                reply_text=self._closing_message(exit_decision.reason),
                meta={"exit": exit_decision.model_dump()},
            )

        scheduling_decision = self.scheduling_advisor.decide(history_text, candidate_message, conversation_now)
        if scheduling_decision.should_schedule_now:
            if scheduling_decision.confirmed_slot:
                self.repository.book_slot(scheduling_decision.confirmed_slot.schedule_id)
                reply_text = self._confirmation_message(scheduling_decision.confirmed_slot)
            else:
                reply_text = self._proposal_message(scheduling_decision.proposed_slots)
            return AgentDecision(
                action="schedule",
                reply_text=reply_text,
                meta={"scheduling": scheduling_decision.model_dump()},
            )

        info_reply = self.info_advisor.respond(history_text, candidate_message)
        return AgentDecision(
            action="continue",
            reply_text=info_reply.reply_text,
            meta={"info": info_reply.model_dump()},
        )

    @staticmethod
    def _closing_message(reason: str) -> str:
        wrap_keywords = ("confirm", "interview", "schedule", "booked")
        if any(k in reason.lower() for k in wrap_keywords):
            return "Great, your interview is confirmed. You'll receive a calendar invite shortly."
        return "No worries — I appreciate the update. Take care!"

    @staticmethod
    def _confirmation_message(slot: ProposedSlot) -> str:
        return f"Great, you're confirmed for {_format_slot(slot)}! You'll receive a calendar invite shortly."

    @staticmethod
    def _proposal_message(slots: list[ProposedSlot]) -> str:
        if not slots:
            return "I don't see any open interview slots right now — I'll follow up once one opens."
        options = ", ".join(_format_slot(s) for s in slots)
        return f"I can offer the following interview times: {options}. Which works best?"
