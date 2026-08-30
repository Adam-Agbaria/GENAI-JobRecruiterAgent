"""Fast unit tests: no OpenAI API calls. Advisors are mocked so these test the
repository's SQL logic and the Main Agent's fusion/priority rules in isolation."""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from app.modules.main_agent.m_1 import MainAgent
from app.modules.scheduling_advisor.m_3 import ProposedSlot, SchedulingDecision
from app.modules.scheduling_db.repository import ScheduleRepository
from app.modules.scheduling_db.schema_migrate import create_and_seed
from app.modules.exit_advisor.m_2 import ExitDecision
from app.modules.info_advisor.m_4 import InfoReply
from datetime import date


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


def _make_agent(repo, exit_should_end=False, exit_confidence=0.9, scheduling_should_schedule=False,
                 confirmed_slot=None, proposed_slots=None, info_reply_text="Sure, tell me more."):
    exit_advisor = MagicMock()
    exit_advisor.decide.return_value = ExitDecision(should_end=exit_should_end, confidence=exit_confidence, reason="test reason")

    scheduling_advisor = MagicMock()
    scheduling_advisor.decide.return_value = SchedulingDecision(
        should_schedule_now=scheduling_should_schedule,
        proposed_slots=proposed_slots or [],
        confirmed_slot=confirmed_slot,
    )

    info_advisor = MagicMock()
    info_advisor.respond.return_value = InfoReply(reply_text=info_reply_text, wants_to_schedule_hint=False)

    return MainAgent(exit_advisor, scheduling_advisor, info_advisor, repo)


def test_end_takes_priority_over_scheduling_and_info(repo):
    agent = _make_agent(repo, exit_should_end=True, exit_confidence=0.9, scheduling_should_schedule=True)
    decision = agent.step([], "I'm no longer interested.", datetime.now())
    assert decision.action == "end"


def test_low_confidence_end_does_not_trigger(repo):
    agent = _make_agent(repo, exit_should_end=True, exit_confidence=0.2, scheduling_should_schedule=False)
    decision = agent.step([], "hello", datetime.now())
    assert decision.action != "end"


def test_schedule_beats_continue_when_proposing(repo):
    slot = ProposedSlot(schedule_id=1, date="2024-04-05", time="10:00:00", position="Python Dev")
    agent = _make_agent(repo, scheduling_should_schedule=True, proposed_slots=[slot])
    decision = agent.step([], "I'd like to schedule.", datetime.now())
    assert decision.action == "schedule"
    assert "2024-04-05" in decision.reply_text


def test_schedule_confirmation_books_the_slot(repo):
    slots = repo.find_available_slots("Python Dev", datetime(2024, 1, 1), None, limit=1)
    real_slot_id = slots[0].schedule_id
    confirmed = ProposedSlot(schedule_id=real_slot_id, date=slots[0].date, time=slots[0].time, position="Python Dev")

    agent = _make_agent(repo, scheduling_should_schedule=True, confirmed_slot=confirmed)
    decision = agent.step([], "Monday at 3 PM works.", datetime.now())

    assert decision.action == "schedule"
    remaining = repo.find_available_slots("Python Dev", datetime(2024, 1, 1), None, limit=100)
    assert all(s.schedule_id != real_slot_id for s in remaining)


def test_falls_through_to_info_advisor_when_no_end_or_schedule(repo):
    agent = _make_agent(repo, info_reply_text="Tell me about your Python experience.")
    decision = agent.step([], "Hi", datetime.now())
    assert decision.action == "continue"
    assert decision.reply_text == "Tell me about your Python experience."
