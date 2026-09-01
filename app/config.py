"""Shared configuration: environment loading, model names, and file paths."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

SCHEDULE_DB_PATH = str(PROJECT_ROOT / os.getenv("SCHEDULE_DB_PATH", "./data/schedule.db"))
CHROMA_DIR = str(PROJECT_ROOT / os.getenv("CHROMA_DIR", "./chroma_store"))
SCHEDULE_LOOKAHEAD_DAYS = int(os.getenv("SCHEDULE_LOOKAHEAD_DAYS", "180"))

JOB_DESCRIPTION_PDF = str(PROJECT_ROOT / "Python Developer Job Description.pdf")
DEFAULT_POSITION = "Python Dev"
