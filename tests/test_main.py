"""Fast unit tests: no OpenAI API calls. Advisors are mocked so these test the
repository's SQL logic and each Main Agent advisor-tool wrapper in isolation.

The Main Agent's *routing* (which advisor to consult, and when to finalize) is
itself an LLM decision now (see app/modules/main_agent/m_1.py), so it can't be
tested deterministically without a real API call. What CAN be tested without
one is that each tool wrapper calls the right advisor and produces the right
side effects (e.g. booking on confirmation) — that's what these tests cover.
A single real end-to-end test of MainAgent.step() is included, gated on an
OpenAI API key being configured."""
from datetime import date, datetime
from unittest.mock import MagicMock

import pytest

from app import config
from app.modules.main_agent.m_1 import MainAgent
from app.modules.scheduling_advisor.m_3 import ProposedSlot, SchedulingDecision
from app.modules.scheduling_db.repository import ScheduleRepository
from app.modules.scheduling_db.schema_migrate import create_and_seed
from app.modules.exit_advisor.m_2 import ExitDecision
from app.modules.info_advisor.m_4 import InfoReply


@pytest.fixture()
def repo(tmp_path):
    db_path = str(tmp_path / "schedule.db")
    create_and_seed(db_path, start_date=date(2024, 1, 1), horizon_days=30, seed=42)
    return ScheduleRepository(db_path)


def test_find_available_slots_returns_sorted_results(repo):
    slots = repo.find_available_slots("Python Dev", datetime(2024, 1, 1), None, limit=3)
    assert len(slots) <= 3
    dates_times = [(s.date, s.time) for s in slots]
    assert dates_times == sorted(dates_times)
    assert all(s.position == "Python Dev" for s in slots)


def test_book_slot_flips_availability_and_is_idempotent(repo):
    slots = repo.find_available_slots("Python Dev", datetime(2024, 1, 1), None, limit=1)
    assert slots, "fixture seed should produce at least one available slot in 30 days"
    slot_id = slots[0].schedule_id

    assert repo.book_slot(slot_id) is True
    assert repo.book_slot(slot_id) is False  # already booked, no row updated

    remaining = repo.find_available_slots("Python Dev", datetime(2024, 1, 1), None, limit=100)
    assert all(s.schedule_id != slot_id for s in remaining)


def _make_agent(repo, exit_advisor=None, scheduling_advisor=None, info_advisor=None):
    return MainAgent(
        exit_advisor or MagicMock(),
        scheduling_advisor or MagicMock(),
        info_advisor or MagicMock(),
        repo,
    )


def test_consult_exit_advisor_tool_delegates_and_returns_decision(repo):
    exit_advisor = MagicMock()
    exit_advisor.decide.return_value = ExitDecision(
        should_end=True, confidence=0.9, end_reason_category="disinterest", reason="no longer interested"
    )
    agent = _make_agent(repo, exit_advisor=exit_advisor)

    tools, tool_by_name = agent._build_tools("history", "I'm no longer interested.", datetime.now())
    result = tool_by_name["consult_exit_advisor"].invoke({})

    exit_advisor.decide.assert_called_once_with("history", "I'm no longer interested.")
    assert result["should_end"] is True
    assert result["end_reason_category"] == "disinterest"


def test_consult_scheduling_advisor_tool_proposes_real_slots(repo):
    slot = ProposedSlot(schedule_id=1, date="2024-04-05", time="10:00:00", position="Python Dev")
    scheduling_advisor = MagicMock()
    scheduling_advisor.decide.return_value = SchedulingDecision(
        should_schedule_now=True, proposed_slots=[slot], confirmed_slot=None
    )
    agent = _make_agent(repo, scheduling_advisor=scheduling_advisor)
    now = datetime.now()

    tools, tool_by_name = agent._build_tools("history", "how about next Friday?", now)
    result = tool_by_name["consult_scheduling_advisor"].invoke({})

    scheduling_advisor.decide.assert_called_once_with("history", "how about next Friday?", now)
    assert result["proposed_slots"][0]["date"] == "2024-04-05"


def test_consult_scheduling_advisor_tool_books_confirmed_slot(repo):
    real_slot = repo.find_available_slots("Python Dev", datetime(2024, 1, 1), None, limit=1)[0]
    confirmed = ProposedSlot(
        schedule_id=real_slot.schedule_id, date=real_slot.date, time=real_slot.time, position="Python Dev"
    )
    scheduling_advisor = MagicMock()
    scheduling_advisor.decide.return_value = SchedulingDecision(
        should_schedule_now=True, proposed_slots=[], confirmed_slot=confirmed
    )
    agent = _make_agent(repo, scheduling_advisor=scheduling_advisor)

    tools, tool_by_name = agent._build_tools("history", "Monday at 3 PM works.", datetime.now())
    result = tool_by_name["consult_scheduling_advisor"].invoke({})

    assert result["confirmed_slot"]["schedule_id"] == real_slot.schedule_id
    remaining_ids = {
        s.schedule_id for s in repo.find_available_slots("Python Dev", datetime(2024, 1, 1), None, limit=1000)
    }
    assert real_slot.schedule_id not in remaining_ids


def test_consult_info_advisor_tool_delegates_and_returns_reply(repo):
    info_advisor = MagicMock()
    info_advisor.respond.return_value = InfoReply(
        reply_text="Tell me about your Python experience.", wants_to_schedule_hint=False
    )
    agent = _make_agent(repo, info_advisor=info_advisor)

    tools, tool_by_name = agent._build_tools("history", "Hi", datetime.now())
    result = tool_by_name["consult_info_advisor"].invoke({})

    info_advisor.respond.assert_called_once_with("history", "Hi")
    assert result["reply_text"] == "Tell me about your Python experience."


@pytest.mark.skipif(not config.OPENAI_API_KEY, reason="requires a real OpenAI API key")
def test_step_end_to_end_routes_to_continue_with_real_advisors(repo):
    from app.modules.exit_advisor.m_2 import ConversationExitAdvisor
    from app.modules.scheduling_advisor.m_3 import InterviewSchedulingAdvisor

    class _StubInfoAdvisor:
        def respond(self, history_text, candidate_message):
            return InfoReply(reply_text="Could you tell me more about your experience?", wants_to_schedule_hint=False)

    agent = MainAgent(
        exit_advisor=ConversationExitAdvisor(),
        scheduling_advisor=InterviewSchedulingAdvisor(repo, position="Python Dev"),
        info_advisor=_StubInfoAdvisor(),
        repository=repo,
    )
    decision = agent.step([], "hey, nice to meet you", datetime(2024, 4, 3, 15, 12, 0))
    assert decision.action == "continue"
    assert decision.reply_text
