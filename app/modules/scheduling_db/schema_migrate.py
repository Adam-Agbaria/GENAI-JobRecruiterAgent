"""
Builds and seeds a SQLite `schedule` table that mirrors the schema and
business rules of the original db_Tech.sql (SQL Server), adapted for a
portable course PoC:

- Original: dbo.Schedule(ScheduleID, date, time, position, available BIT),
  seeded for all of calendar year 2024, Tue/Wed/Thu/Fri/Sun only,
  hours 09:00-17:00 hourly, positions {Python Dev, Sql Dev, Analyst, ML},
  availability ~50/50 via a pseudo-normal CHECKSUM(NEWID()) threshold.
- Here: same weekday filter, hour range, and position list, same ~50/50
  availability, but the date range is anchored to a configurable
  `start_date` + `horizon_days` window instead of being hardcoded to 2024,
  so live-demo relative-date queries ("next Friday") always resolve
  against real, non-expired data. The evaluation notebook re-seeds a
  separate DB anchored at 2024-01-01 to stay faithful to the labeled
  dataset's timestamps.
"""
import random
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

DEFAULT_POSITIONS = ["Python Dev", "Sql Dev", "Analyst", "ML"]
EXCLUDED_WEEKDAYS = {"Saturday", "Monday"}  # matches db_Tech.sql: Tue-Fri & Sun only
HOURS = list(range(9, 18))  # 09:00 .. 17:00 inclusive


@dataclass
class SeedStats:
    rows_inserted: int
    start_date: date
    end_date: date


def _valid_dates(start_date: date, horizon_days: int):
    for offset in range(horizon_days + 1):
        d = start_date + timedelta(days=offset)
        if d.strftime("%A") not in EXCLUDED_WEEKDAYS:
            yield d


def create_and_seed(
    db_path: str,
    start_date: date | None = None,
    horizon_days: int = 180,
    positions: list[str] | None = None,
    seed: int | None = None,
) -> SeedStats:
    """(Re)creates the `schedule` table and seeds it with randomized availability."""
    start_date = start_date or date.today()
    positions = positions or DEFAULT_POSITIONS
    rng = random.Random(seed)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS schedule")
        conn.execute(
            """
            CREATE TABLE schedule (
                schedule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT NOT NULL,
                time        TEXT NOT NULL,
                position    TEXT NOT NULL,
                available   INTEGER NOT NULL
            )
            """
        )

        rows = []
        end_date = start_date
        for d in _valid_dates(start_date, horizon_days):
            end_date = d
            for hour in HOURS:
                for position in positions:
                    available = 1 if rng.random() >= 0.5 else 0
                    rows.append((d.isoformat(), f"{hour:02d}:00:00", position, available))

        conn.executemany(
            "INSERT INTO schedule (date, time, position, available) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
        return SeedStats(rows_inserted=len(rows), start_date=start_date, end_date=end_date)
    finally:
        conn.close()


if __name__ == "__main__":
    from app import config

    stats = create_and_seed(
        db_path=config.SCHEDULE_DB_PATH,
        start_date=date.today(),
        horizon_days=config.SCHEDULE_LOOKAHEAD_DAYS,
    )
    print(
        f"Seeded {stats.rows_inserted} rows into {config.SCHEDULE_DB_PATH} "
        f"covering {stats.start_date} .. {stats.end_date}"
    )
