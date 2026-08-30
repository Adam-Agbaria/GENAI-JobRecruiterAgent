"""
Interview Scheduling Advisor.

Decides whether now is the right time to propose or confirm an interview
slot. Resolves relative date language ("next Friday") against the
conversation's own reference timestamp, then calls a bound tool that queries
the SQLite-backed schedule for real availability (function calling).

Two-pass design:
  1. A tool-calling pass lets the LLM resolve the date/time bounds itself
     (a language-understanding task) and invoke `get_available_slots`
     (a deterministic DB lookup, kept out of the LLM's hands).
  2. A structured-output pass turns the conversation + tool result into a
     final SchedulingDecision the Main Agent can act on directly.
"""
import json
from datetime import datetime
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app import config
from app.modules.scheduling_db.repository import ScheduleRepository

SYSTEM_PROMPT = """You are the Interview Scheduling Advisor for a recruiting SMS bot hiring \
for the "{position}" position. Your job is to decide whether now is the right time to \
propose or confirm an interview slot, and if so, to look up real availability.

The conversation's reference "now" is {reference_iso}. When the candidate uses relative \
date language, resolve it against this reference before calling the tool.

Worked example: if the reference is Wednesday 2024-04-03T15:12:00Z and the candidate says \
"next Friday", the target date is 2024-04-05 (the upcoming Friday after the reference date). \
If they say "Monday at 3 PM" with no explicit date, resolve it to the nearest upcoming Monday \
at 15:00 on or after the reference date.

Call `get_available_slots` whenever:
- You are about to propose new interview times to the candidate, OR
- The candidate just named a specific day/time and you need to verify it's actually \
available and fetch its schedule_id to confirm it (in that case set earliest_iso and \
latest_iso to the same narrow window around the requested time so only that slot matches).

Do NOT call the tool if the candidate hasn't shown any interest in scheduling yet — in that \
case scheduling is not appropriate right now.
"""


class ProposedSlot(BaseModel):
    schedule_id: int
    date: str
    time: str
    position: str


class SchedulingDecision(BaseModel):
    should_schedule_now: bool = Field(
        description="True if a slot should be proposed or has just been confirmed"
    )
    proposed_slots: list[ProposedSlot] = Field(
        default_factory=list, description="Up to 3 candidate slots being offered, nearest first"
    )
    confirmed_slot: Optional[ProposedSlot] = Field(
        default=None, description="Set only when the candidate has just accepted a specific slot"
    )


class GetAvailableSlotsArgs(BaseModel):
    reference_datetime_iso: str = Field(
        description="The conversation's current timestamp in ISO-8601, used as 'now'"
    )
    relative_phrase: str = Field(
        description="The raw relative-date phrase from the candidate, e.g. 'next Friday', or '' if absolute"
    )
    earliest_iso: str = Field(
        description="Resolved absolute earliest datetime bound in ISO-8601"
    )
    latest_iso: Optional[str] = Field(default=None, description="Resolved absolute latest datetime bound in ISO-8601")
    limit: int = Field(default=3, description="Max number of slots to return")


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


class InterviewSchedulingAdvisor:
    def __init__(self, repository: ScheduleRepository, position: str = config.DEFAULT_POSITION):
        self.repository = repository
        self.position = position

        self._slots_tool = StructuredTool.from_function(
            func=self._get_available_slots,
            name="get_available_slots",
            description=(
                "Look up available interview slots for the target position within a "
                "date/time range. Resolve any relative date phrase into concrete ISO-8601 "
                "bounds using reference_datetime_iso as 'now' before calling."
            ),
            args_schema=GetAvailableSlotsArgs,
        )

        self._tool_calling_llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.1
        ).bind_tools([self._slots_tool])

        self._structured_llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL, api_key=config.OPENAI_API_KEY, temperature=0.1
        ).with_structured_output(SchedulingDecision)

    def _get_available_slots(
        self,
        reference_datetime_iso: str,
        relative_phrase: str,
        earliest_iso: str,
        latest_iso: str | None = None,
        limit: int = 3,
    ) -> list[dict]:
        earliest = _parse_iso(earliest_iso)
        latest = _parse_iso(latest_iso) if latest_iso else None
        slots = self.repository.find_available_slots(self.position, earliest, latest, limit)
        return [s.as_dict() for s in slots]

    def decide(self, history_text: str, candidate_message: str, conversation_now: datetime) -> SchedulingDecision:
        reference_iso = conversation_now.isoformat()
        system = SYSTEM_PROMPT.format(position=self.position, reference_iso=reference_iso)
        messages = [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Conversation so far:\n{history_text}\n\n"
                    f"Candidate's latest message: {candidate_message}"
                )
            ),
        ]

        ai_msg = self._tool_calling_llm.invoke(messages)
        messages.append(ai_msg)

        for tool_call in getattr(ai_msg, "tool_calls", None) or []:
            if tool_call["name"] == "get_available_slots":
                result = self._slots_tool.invoke(tool_call["args"])
                messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tool_call["id"]))

        messages.append(HumanMessage(content="Based on the above, finalize your scheduling decision now."))
        return self._structured_llm.invoke(messages)
