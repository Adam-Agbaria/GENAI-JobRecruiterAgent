"""
Conversation Exit Advisor.

Decides whether the conversation should end at this point — either because
the candidate is disengaging (no longer interested, found another job, asked
to stop being contacted) or because the interview has just been confirmed and
the exchange has naturally wrapped up.

Implemented entirely via prompt engineering (role prompt + explicit
instructions + few-shot examples drawn from sms_conversations.json + a low
temperature API parameter), not fine-tuning: an OpenAI fine-tuning pass was
initially planned for this advisor, but was dropped after an OpenAI API
change made it impractical for this course project, per updated course
guidance to use prompt engineering here instead.
"""
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app import config

SYSTEM_PROMPT = """You are the Conversation Exit Advisor for a recruiting SMS bot hiring a \
Python Developer. Your only job is to decide whether the conversation should END now.

End the conversation when:
- The candidate clearly disengages: no longer interested, found another job, \
asks to stop being contacted or removed from the list.
- The RECRUITER's own prior message in the conversation history already proposed one or more \
specific candidate date/time options (e.g., "Wednesday at 10 AM or Thursday at 2 PM"), AND the \
candidate's latest message is accepting one of those exact times — nothing more to discuss.

Do NOT end the conversation when:
- The candidate is still asking questions, still negotiating a time, or otherwise engaged.
- The candidate is merely expressing general willingness or interest in scheduling (e.g., \
"sure, I'd love to", "next Friday works for me", "sounds great, let's set something up") but \
no recruiter message has yet proposed concrete date/time options for them to accept. This is \
still an open invitation to schedule, not a confirmation — it should be routed to actually \
look up and propose real times, not treated as a wrap-up.

Examples of candidate messages that SHOULD end the conversation (a concrete slot was already \
proposed by the recruiter earlier in the history, and this message accepts it):
- "Please remove me from your list. Thanks." (disinterest — always ends regardless of history)
- "I'm no longer interested in the position." (disinterest — always ends)
- "I will be intouch, please stop texting me" (disinterest, informal phrasing/typos still count)
- "I'm sorry, but I'm no longer interested." (polite disinterest — always ends)
- "Monday at 3 PM is good." (given the recruiter already offered specific slots including \
Monday 3 PM earlier in the history -> accepting one of them, wrap up)
- "Tuesday at 10 AM works." (same pattern: accepting a specific previously-offered slot)
- "Sounds greate, see you then" (acknowledgment right after the recruiter already confirmed \
a booked slot)

Examples of candidate messages that should NOT end the conversation:
- "I've been using Python professionally for five years, mostly for data analysis." \
(still answering questions, engaged)
- "Could you share more about the company's cloud technologies?" (asking a question)
- "I can't at that time, I'm busy." (still negotiating a time, not disengaging)
- "I would like to set an appointment, does Monday at 3 PM work?" (candidate proposing a \
time themselves — nothing was offered by the recruiter yet to accept, and this needs to be \
checked against real availability first)
- "Sure, next Friday works for me." with no prior recruiter message naming specific times \
(general willingness only — no concrete slot exists yet to be confirming; route to scheduling \
so real times can be looked up and offered first)

When should_end is True, also set end_reason_category: "disinterest" for a disengaging \
candidate, or "confirmation" for accepting an already-proposed slot. Leave it "not_applicable" \
when should_end is False.
"""


class ExitDecision(BaseModel):
    should_end: bool = Field(description="True if the conversation should end now")
    confidence: float = Field(description="Confidence in [0, 1]")
    end_reason_category: Literal["disinterest", "confirmation", "not_applicable"] = Field(
        default="not_applicable",
        description=(
            "Why the conversation should end, when should_end is True: 'disinterest' if the "
            "candidate disengaged, 'confirmation' if they just accepted a specific slot the "
            "recruiter already proposed. 'not_applicable' when should_end is False."
        ),
    )
    reason: str = Field(description="One short sentence explaining the decision")


class ConversationExitAdvisor:
    def __init__(self, model: str | None = None):
        self.llm = ChatOpenAI(
            model=model or config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.1,
        ).with_structured_output(ExitDecision)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Conversation so far:\n{history_text}\n\nCandidate's latest message: {candidate_message}"),
            ]
        )

    def decide(self, history_text: str, candidate_message: str) -> ExitDecision:
        chain = self.prompt | self.llm
        return chain.invoke({"history_text": history_text, "candidate_message": candidate_message})
