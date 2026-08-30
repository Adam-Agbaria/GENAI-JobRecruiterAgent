"""Read/write access to the `schedule` table for the Scheduling Advisor."""
import sqlite3
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Slot:
    schedule_id: int
    date: str  # "YYYY-MM-DD"
    time: str  # "HH:MM:SS"
    position: str

    def as_dict(self) -> dict:
        return {
            "schedule_id": self.schedule_id,
            "date": self.date,
            "time": self.time,
            "position": self.position,
        }


class ScheduleRepository:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def find_available_slots(
        self,
        position: str,
        earliest: datetime,
        latest: datetime | None = None,
        limit: int = 3,
    ) -> list[Slot]:
        """Returns the nearest `limit` available slots for `position` at or after
        `earliest` (and, if given, at or before `latest`), sorted by date/time."""
        earliest_date, earliest_time = earliest.strftime("%Y-%m-%d"), earliest.strftime("%H:%M:%S")

        query = (
            "SELECT schedule_id, date, time, position FROM schedule "
            "WHERE position = ? AND available = 1 "
            "AND (date > ? OR (date = ? AND time >= ?))"
        )
        params: list = [position, earliest_date, earliest_date, earliest_time]

        if latest is not None:
            latest_date, latest_time = latest.strftime("%Y-%m-%d"), latest.strftime("%H:%M:%S")
            query += " AND (date < ? OR (date = ? AND time <= ?))"
            params += [latest_date, latest_date, latest_time]

        query += " ORDER BY date ASC, time ASC LIMIT ?"
        params.append(limit)

        conn = self._connect()
        try:
            rows = conn.execute(query, params).fetchall()
            return [Slot(schedule_id=r[0], date=r[1], time=r[2], position=r[3]) for r in rows]
        finally:
            conn.close()

    def book_slot(self, schedule_id: int) -> bool:
        """Marks a slot as no longer available. Returns True if a row was updated."""
        conn = self._connect()
        try:
            cur = conn.execute(
                "UPDATE schedule SET available = 0 WHERE schedule_id = ? AND available = 1",
                (schedule_id,),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
