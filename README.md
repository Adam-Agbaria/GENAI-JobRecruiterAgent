# SMS Job-Candidate Recruiting Bot (GenAI Course Project)

## Purpose

A conversational bot that stands in for a human recruiter texting with candidates for a
**Python Developer** position. Each turn it decides whether to **continue** the conversation,
**schedule** an interview, or **end** the conversation — gathering/verifying candidate
information, answering questions about the role (grounded in the job description via RAG),
and booking a real interview slot against a recruiter availability database.

## Architecture

A **Main Agent** orchestrates the conversation and consults three specialized **Advisors** at
each turn, in priority order:

1. **Conversation Exit Advisor** — decides if the conversation should end now (candidate
   disengaged, or an interview was just confirmed and the exchange is wrapping up).
2. **Interview Scheduling Advisor** — decides if it's time to propose or confirm an interview
   slot. Resolves relative dates ("next Friday") against the conversation's own timestamp, then
   calls a function-calling tool that queries a SQL-backed availability table for real open slots.
3. **Conversation Info Advisor** — answers the candidate's questions using retrieval-augmented
   generation over the job description (Chroma vector store), and drafts the next message.

The Main Agent is a deterministic Python function (not an autonomous LLM-driven control loop),
so its decisions are unit-testable and reproducible when replayed against the labeled evaluation
set. Each Advisor internally uses LangChain (chat models, prompt templates, structured output,
tool-calling, and a Chroma retriever).

## Project Structure

```
app/
  main.py                     # CLI entry point + build_main_agent() shared wiring
  config.py                   # env loading, model names, file paths
  conversation.py             # shared Turn/history formatting helpers
  modules/
    main_agent/m_1.py         # MainAgent: orchestration + fusion logic
    exit_advisor/m_2.py       # Conversation Exit Advisor
    scheduling_advisor/m_3.py # Interview Scheduling Advisor + function-calling tool
    info_advisor/m_4.py       # Conversation Info Advisor (RAG)
    scheduling_db/            # SQLite schema/seed + repository (see "SQL backend" below)
    embedding/build_index.py  # offline PDF -> Chroma pipeline
    fine_tuning/              # Exit Advisor fine-tuning pipeline
streamlit_app/
  streamlit_main.py           # Streamlit chat UI (PoC in place of real SMS)
  utils.py                    # UI-only helpers, no business logic
tests/
  test_main.py                # unit tests (repository + Main Agent fusion, mocked advisors)
  test_evals.ipynb            # accuracy + confusion matrix against sms_conversations.json
data/schedule.db              # generated SQLite DB (committed — see below)
chroma_store/                 # generated Chroma vector store (committed — see below)
```

### Deviations from the literal assignment skeleton (and why)

- **SQL Server → SQLite.** The provided `db_Tech.sql` targets SQL Server, which isn't practical
  for a portable course PoC deployed on Streamlit Community Cloud (no SQL Server available there).
  `app/modules/scheduling_db/schema_migrate.py` reimplements the same schema and business rules
  (Tue/Wed/Thu/Fri/Sun only, 09:00-17:00 hourly, ~50/50 random availability) in SQLite, with a
  dynamic date-seeding window (anchored to "today" + a lookahead) instead of the original's
  hardcoded 2024, so relative-date queries in the live demo always have real matching data. The
  evaluation notebook re-seeds a separate DB anchored at 2024-01-01 to stay faithful to the
  labeled dataset's timestamps.
- **Descriptive module names** (`main_agent`, `exit_advisor`, ...) instead of literal
  `module_1`/`module_2` — the assignment spec itself calls those illustrative example names.
- **`app/config.py` and `app/conversation.py` added** — shared env loading and history-formatting
  helpers reused by every advisor, the CLI, the Streamlit app, and the eval notebook.
- **`chroma_store/` and `data/schedule.db` are committed**, not gitignored — Streamlit Community
  Cloud's filesystem is ephemeral on cold start, and rebuilding them on every boot would waste
  embedding API calls and add latency for a tiny, rarely-changing source PDF.
- **Plain Python orchestration instead of LangGraph/AgentExecutor** for the Main Agent —
  deterministic, unit-testable turn-by-turn decisions are required to score accuracy/confusion
  matrix against the labeled dataset; LangChain is still used inside every advisor.

## Install & Run Locally

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cp .env.example .env          # then fill in OPENAI_API_KEY
```

Seed the scheduling database and build the job-description vector index (both one-time,
offline steps):

```bash
python -m app.modules.scheduling_db.schema_migrate
python -m app.modules.embedding.build_index
```

Run the CLI demo (type candidate replies, `quit` to exit):

```bash
python -m app.main
```

Run the Streamlit PoC:

```bash
streamlit run streamlit_app/streamlit_main.py
```

Run unit tests (fast, no API calls — advisors are mocked):

```bash
pytest tests/test_main.py
```

Run the evaluation notebook (`tests/test_evals.ipynb`) to reproduce accuracy + confusion matrix
against the 59 labeled turns in `sms_conversations.json`.

## Usage Example

```
Recruiter: Thanks for applying to our Python Developer opening. What kinds of Python
           projects have you worked on recently?
Candidate: I've been using Python professionally for five years, mostly for data analysis.
Recruiter: That's great experience! Would you be open to scheduling a quick interview?
Candidate: Sure, how about next Friday?
Recruiter: I can offer the following interview times: 2026-09-04 at 09:00, 2026-09-04 at
           11:00, 2026-09-04 at 14:00. Which works best?
Candidate: 9am works.
Recruiter: Great, you're confirmed for 2026-09-04 at 09:00! You'll receive a calendar
           invite shortly.
```

## Fine-Tuning Note

`app/modules/fine_tuning/` builds a real OpenAI fine-tuning pipeline for the Exit Advisor
(`prepare_dataset.py` → `run_finetune.py`), and it was run as a genuine attempt. However,
`sms_conversations.json` only contains **15 positive ("end") examples across 59 labeled turns**
— a very small set for a robust fine-tune. Because of this, the Exit Advisor's default and
fallback path is careful **few-shot prompting** (`app/modules/exit_advisor/m_2.py`), and the
fine-tuned model is only used when `EXIT_ADVISOR_FINE_TUNED_MODEL_ID` is explicitly set in
`.env` — no code changes are needed to switch between the two.

## Evaluation Summary

See `tests/test_evals.ipynb` for the full accuracy and confusion-matrix results, run against all
59 labeled recruiter turns in `sms_conversations.json` (25 `continue`, 19 `schedule`, 15 `end`).
Run the notebook top-to-bottom to reproduce the numbers — results depend on live OpenAI API
calls, so an `OPENAI_API_KEY` must be set in `.env` first.

## Scope Note

Per the assignment brief, this implementation is intentionally simplified: a real production bot
would handle additional edge cases and options beyond what's modeled here.
