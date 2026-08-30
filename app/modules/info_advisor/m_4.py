"""
Conversation Info Advisor.

Answers the candidate's questions about the Python Developer role (grounded
in the job description via a Chroma retriever), drafts the next outbound
message, and nudges the conversation toward scheduling an interview once
enough information has been exchanged.
"""
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from app import config

SYSTEM_PROMPT = """You are a friendly recruiter assistant texting with a candidate for a \
Python Developer role. Keep replies short (1-3 sentences), warm, and SMS-appropriate.

Use ONLY the job description context below to answer factual questions about the role. \
If the answer isn't in the context, say you'll follow up with the recruiter rather than \
guessing.

Job description context:
{context}

After a reasonable amount of back-and-forth (the candidate has shared their background \
and/or gotten their questions answered), gently steer the conversation toward scheduling \
an interview. Don't be pushy on the very first message.

Examples of good replies:
- "Thanks for sharing! We currently deploy to AWS using Docker and ECS. Would you be open \
to scheduling a quick interview with our engineering manager?"
- "Great question — could you tell me a bit about your recent Python projects?"
"""


class InfoReply(BaseModel):
    reply_text: str = Field(description="The next SMS message to send the candidate")
    wants_to_schedule_hint: bool = Field(
        description="True if this reply suggests it's a good time to try scheduling an interview"
    )


class ConversationInfoAdvisor:
    def __init__(self, chroma_dir: str | None = None, k: int = 4):
        embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY)
        self._vectorstore = Chroma(
            persist_directory=chroma_dir or config.CHROMA_DIR,
            embedding_function=embeddings,
        )
        self._retriever = self._vectorstore.as_retriever(search_kwargs={"k": k})

        self.llm = ChatOpenAI(
            model=config.OPENAI_CHAT_MODEL,
            api_key=config.OPENAI_API_KEY,
            temperature=0.5,
        ).with_structured_output(InfoReply)

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Conversation so far:\n{history_text}\n\nCandidate's latest message: {candidate_message}"),
            ]
        )

    def respond(self, history_text: str, candidate_message: str) -> InfoReply:
        docs = self._retriever.invoke(candidate_message or history_text)
        context = "\n\n".join(d.page_content for d in docs) or "(no context retrieved)"

        chain = self.prompt | self.llm
        return chain.invoke(
            {
                "history_text": history_text,
                "candidate_message": candidate_message,
                "context": context,
            }
        )
