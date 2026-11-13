"""Session memory: LangGraph checkpointer on SQLite.

Every turn's state (messages, last plan, evidence) is saved per thread, so a
conversation survives multiple turns and API restarts.

Thread isolation: the key is "<user_id>:<thread_id>". The user id comes from the
JWT, so one user can never load another user's conversation, even by guessing
a thread id.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import get_settings


@asynccontextmanager
async def open_checkpointer():
    path = Path(get_settings().memory_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


def thread_key(user_id: str, thread_id: str) -> str:
    return f"{user_id}:{thread_id}"
