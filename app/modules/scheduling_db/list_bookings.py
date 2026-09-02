"""Lists interview slots the app has actually booked (via ScheduleRepository.book_slot),
distinct from slots that are simply unavailable due to randomly-seeded pre-existing
commitments.

Usage:
    python -m app.modules.scheduling_db.list_bookings
"""
from app import config
from app.modules.scheduling_db.repository import ScheduleRepository

if __name__ == "__main__":
    bookings = ScheduleRepository(config.SCHEDULE_DB_PATH).list_bookings()
    if not bookings:
        print("No booked interview slots yet.")
    else:
        print(f"{'schedule_id':<12}{'date':<12}{'time':<10}{'position':<12}booked_at_utc")
        for b in bookings:
            print(f"{b['schedule_id']:<12}{b['date']:<12}{b['time']:<10}{b['position']:<12}{b['booked_at_utc']}")
