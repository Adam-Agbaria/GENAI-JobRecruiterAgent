"""
Main Agent: orchestrates the conversation turn-by-turn.

Per the assignment's workflow diagram, the Main Agent dynamically decides
which ONE advisor to consult at a time — it is not a fixed call-all-three
sequence. It may consult more than one advisor in sequence within a single
turn ("Consult Advisor Again") before finalizing its reply. This is
implemented as a bounded LangChain tool-calling loop: each advisor is
exposed to the Main Agent's LLM as a tool it can choose to invoke (each of
which internally "processes the complete chat history" and returns a
domain-specific decision — end/don't-end, sched/don't-sched with a real SQL
lookup, info-needed/not-needed with a real vector lookup). Once the Main
Agent has consulted enough advisors, a second structured-output pass
finalizes this turn's action + reply text.
"""
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app import config
from app.conversation import format_history
from app.modules.exit_advisor.m_2 import ConversationExitAdvisor
from app.modules.info_advisor.m_4 import ConversationInfoAdvisor
from app.modules.scheduling_advisor.m_3 import InterviewSchedulingAdvisor
from app.modules.scheduling_db.repository import ScheduleRepository

Action = Literal["continue", "schedule", "end"]

MAX_CONSULTATIONS = 4  # 3 advisors + one buffer round, to bound the loop's cost/latency


@dataclass
class AgentDecision:
    action: Action
    reply_text: str
    meta: dict = field(default_factory=dict)


class FinalTurnDecision(BaseModel):
    action: Action = Field(description="The single action to take for this turn")
    reply_text: str = Field(description="The exact SMS message to send the candidate")


SYSTEM_PROMPT = """You are the Main Agent for a recruiting SMS bot hiring for the \
"{position}" position. You orchestrate the conversation with a candidate. You have three \
specialist advisors, and you consult them ONE AT A TIME by calling the matching tool:

- consult_exit_advisor: whether the conversation should end now (candidate disengaged, or \
just accepted a previously-offered slot and the exchange is wrapping up).
- consult_scheduling_advisor: whether now is the right time to propose or confirm a real \
interview slot (it will look up actual availability if appropriate).
- consult_info_advisor: how to answer the candidate's question or draft the next message, \
grounded in the job description.

Required order of consideration, every turn: first consult_exit_advisor (in case the \
candidate is disengaging, or just confirmed a slot). If it does not end the conversation, \
you MUST then consult consult_scheduling_advisor next (in case it's time to propose or \
confirm a slot) — do not skip straight to the info advisor just because the candidate shared \
some background; let the scheduling advisor itself decide whether that's enough to propose \
times. Only if consult_scheduling_advisor also says now is not the right time should you \
consult consult_info_advisor. Call at most one tool per step, in this order, until you reach \
one that changes the action for this turn.

Once you have consulted enough advisors, stop and finalize the turn: choose exactly one \
action (continue, schedule, or end) and write the exact SMS reply text to send the candidate.

CRITICAL: never finalize action "end" on the grounds that a slot was confirmed unless you \
have consulted consult_scheduling_advisor THIS turn and its result includes a non-null \
confirmed_slot — always state the exact date/time from that confirmed_slot in your reply, \
never a time the candidate merely mentioned. If consult_exit_advisor reports \
end_reason_category "confirmation", you must still consult consult_scheduling_advisor \
before finalizing, to actually verify and book the specific slot. If the candidate names a \
time that doesn't match any slot the recruiter actually offered, do not confirm it as booked \
— consult consult_scheduling_advisor to check real availability for that time, or ask the \
candidate to clarify instead.
"""


class MainAgent:
    def __init__(
        self,
        exit_advisor: ConversationExitAdvisor,
        scheduling_advisor: InterviewSchedulingAdvisor,
        info_advisor: ConversationInfoAdvisor,
        repository: ScheduleRepository,
        position: str = config.DEFAULT_POSITION,
        model: str | None = None,
    ):
        self.exit_advisor = exit_advisor
        self.scheduling_advisor = scheduling_advisor
        self.info_advisor = info_advisor
        self.repository = repository
        self.position = position
        self._model_name = model or config.OPENAI_CHAT_MODEL

    def step(self, conversation_history: list, candidate_message: str | None, conversation_now: datetime) -> AgentDecision:
        history_text = format_history(conversation_history)
        candidate_message = candidate_message or "(conversation just started)"

        tools, tool_by_name = self._build_tools(history_text, candidate_message, conversation_now)
        router_llm = ChatOpenAI(
            model=self._model_name, api_key=config.OPENAI_API_KEY, temperature=0.1
        ).bind_tools(tools)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT.format(position=self.position)),
            HumanMessage(
                content=(
                    f"Conversation so far:\n{history_text}\n\n"
                    f"Candidate's latest message: {candidate_message}"
                )
            ),
        ]

        consultations = []
        for _ in range(MAX_CONSULTATIONS):
            ai_msg = router_llm.invoke(messages)
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if not tool_calls:
                break

            for tool_call in tool_calls:
                tool = tool_by_name[tool_call["name"]]
                result = tool.invoke(tool_call["args"])
                consultations.append({"advisor": tool_call["name"], "output": result})
                messages.append(
                    ToolMessage(content=json.dumps(result, default=str), tool_call_id=tool_call["id"])
                )

        messages.append(
            HumanMessage(content="Finalize your decision now: choose the action and write the reply text.")
        )
        finalizer = ChatOpenAI(
            model=self._model_name, api_key=config.OPENAI_API_KEY, temperature=0.1
        ).with_structured_output(FinalTurnDecision)
        final = finalizer.invoke(messages)

        return AgentDecision(action=final.action, reply_text=final.reply_text, meta={"consultations": consultations})

    def _build_tools(self, history_text: str, candidate_message: str, conversation_now: datetime):
        """Wraps each advisor as a tool the router LLM can choose to call. Closed over
        this turn's context so the LLM only ever needs to decide *which* advisor to
        consult, never re-supply the conversation itself."""

        def consult_exit_advisor() -> dict:
            """Ask the Conversation Exit Advisor whether the conversation should end now."""
            decision = self.exit_advisor.decide(history_text, candidate_message)
            return decision.model_dump()

        def consult_scheduling_advisor() -> dict:
            """Ask the Interview Scheduling Advisor whether to propose or confirm a real interview slot."""
            decision = self.scheduling_advisor.decide(history_text, candidate_message, conversation_now)
            if decision.confirmed_slot:
                self.repository.book_slot(decision.confirmed_slot.schedule_id)
            return decision.model_dump()

        def consult_info_advisor() -> dict:
            """Ask the Conversation Info Advisor how to answer the candidate and draft the next reply."""
            reply = self.info_advisor.respond(history_text, candidate_message)
            return reply.model_dump()

        tools = [
            StructuredTool.from_function(
                func=consult_exit_advisor,
                name="consult_exit_advisor",
                description="Ask the Conversation Exit Advisor whether the conversation should end now.",
            ),
            StructuredTool.from_function(
                func=consult_scheduling_advisor,
                name="consult_scheduling_advisor",
                description="Ask the Interview Scheduling Advisor whether to propose or confirm a real interview slot.",
            ),
            StructuredTool.from_function(
                func=consult_info_advisor,
                name="consult_info_advisor",
                description="Ask the Conversation Info Advisor how to answer the candidate and draft the next reply.",
            ),
        ]
        return tools, {t.name: t for t in tools}
