"""
Conversation Exit Advisor.

Decides whether the conversation should end at this point — either because
the candidate is disengaging (no longer interested, found another job, asked
to stop being contacted) or because the interview has just been confirmed and
the exchange has naturally wrapped up.

Uses `EXIT_ADVISOR_MODEL` from app.config, which is the fine-tuned model id
once one is trained (see app/modules/fine_tuning), and otherwise falls back
to the base chat model with the few-shot examples baked into the prompt
below. No other code needs to change when a fine-tuned model becomes
available.
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app import config

SYSTEM_PROMPT = """You are the Conversation Exit Advisor for a recruiting SMS bot hiring a \
Python Developer. Your only job is to decide whether the conversation should END now.

End the conversation when:
- The candidate clearly disengages: no longer interested, found another job, \
asks to stop being contacted or removed from the list.
- An interview slot was just confirmed and the candidate's latest message is simply \
acknowledging/confirming it (nothing more to discuss).

Do NOT end the conversation when the candidate is still asking questions, still \
negotiating a time, or otherwise engaged but nothing has been confirmed yet.

Examples of candidate messages that SHOULD end the conversation:
- "Please remove me from your list. Thanks." (disinterest)
- "I'm no longer interested in the position." (disinterest)
- "I will be in touch, please stop texting me" (disinterest)
- "Monday at 3 PM is good." (right after a slot was already proposed and accepted -> \
interview confirmed, wrap up)
- "Tuesday at 10 AM works." (same pattern: accepting a proposed slot -> wrap up)

Examples of candidate messages that should NOT end the conversation:
- "I've been using Python professionally for five years, mostly for data analysis." \
(still answering questions, engaged)
- "Could you share more about the company's cloud technologies?" (asking a question)
- "I can't at that time, I'm busy." (still negotiating a time, not disengaging)
"""


class ExitDecision(BaseModel):
    should_end: bool = Field(description="True if the conversation should end now")
    confidence: float = Field(description="Confidence in [0, 1]")
    reason: str = Field(description="One short sentence explaining the decision")


class ConversationExitAdvisor:
    def __init__(self, model: str | None = None):
        self.llm = ChatOpenAI(
            model=model or config.EXIT_ADVISOR_MODEL,
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
