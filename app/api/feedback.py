"""Answer-quality feedback: thumbs up / down (+ optional comment) per answer, stored in LangSmith.

Feedback is attached to the LangSmith run of the turn it rates, so every rating sits next to the
full trace of how that answer was produced – which agents ran, what was retrieved, what the model
saw. Filtering runs by `user_score = 0` gives the list of answers to investigate (the feedback loop).

Controls
  - only the user who received the answer may rate it (run ownership is checked);
  - one rating per user and answer: changing the thumb updates it (deterministic feedback id);
  - the comment is length-limited, never logged (it may contain anything), only its length is;
  - LangSmith down or not configured: the rating is still logged and the API says it was not stored.
"""

import asyncio
import uuid
from collections import OrderedDict
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.errors import PermissionDeniedError
from app.logging import get_logger
from app.security.auth import User, get_current_user

router = APIRouter(tags=["feedback"])
log = get_logger(__name__)

FEEDBACK_KEY = "user_score"
LANGSMITH_TIMEOUT = 10.0
_MAX_REMEMBERED = 5000
_FEEDBACK_NS = uuid.UUID("6f3c1c1e-6a52-4f39-9a0e-0d3b6a1f8c11")

# run id -> user id for runs this process served. Lets ownership be checked without a LangSmith call;
# after a restart the run's metadata in LangSmith is used instead.
_RUN_OWNERS: OrderedDict[str, str] = OrderedDict()


def remember_run(run_id: uuid.UUID | str, user_id: str) -> None:
    _RUN_OWNERS[str(run_id)] = user_id
    while len(_RUN_OWNERS) > _MAX_REMEMBERED:
        _RUN_OWNERS.popitem(last=False)


class FeedbackRequest(BaseModel):
    run_id: uuid.UUID
    score: Literal[0, 1] = Field(description="1 = helpful (thumbs up), 0 = not helpful (thumbs down)")
    comment: str | None = Field(default=None, max_length=500)


def _langsmith_client():
    from langsmith import Client

    return Client()


def _owner_from_langsmith(run_id: str) -> str | None:
    run = _langsmith_client().read_run(run_id)
    return ((run.extra or {}).get("metadata") or {}).get("user_id")


def _store(run_id: str, user_id: str, score: int, comment: str | None) -> None:
    from langsmith.utils import LangSmithConflictError

    client = _langsmith_client()
    feedback_id = uuid.uuid5(_FEEDBACK_NS, f"{run_id}:{user_id}")  # same user + answer -> same feedback
    try:
        client.create_feedback(run_id, FEEDBACK_KEY, score=score, comment=comment, feedback_id=feedback_id,
                               source_info={"user_id": user_id}, stop_after_attempt=2)
    except LangSmithConflictError:  # already rated: update the existing rating
        client.update_feedback(feedback_id, score=score, comment=comment)


@router.post("/feedback")
async def feedback(body: FeedbackRequest, user: User = Depends(get_current_user)) -> dict:
    run_id = str(body.run_id)
    owner = _RUN_OWNERS.get(run_id)
    if owner is None:
        try:
            owner = await asyncio.wait_for(asyncio.to_thread(_owner_from_langsmith, run_id), LANGSMITH_TIMEOUT)
        except Exception:  # noqa: BLE001 – unknown run or LangSmith unavailable: cannot prove ownership
            owner = None
    if owner != user.user_id:
        raise PermissionDeniedError("You can only rate your own answers.")

    log.info("answer_feedback", run_id=run_id, score=body.score, comment_chars=len(body.comment or ""))
    try:
        await asyncio.wait_for(asyncio.to_thread(_store, run_id, user.user_id, body.score, body.comment),
                               LANGSMITH_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 – the rating is in the log; tell the UI it is not in LangSmith
        log.warning("feedback_not_stored", run_id=run_id, error_type=type(exc).__name__)
        return {"run_id": run_id, "score": body.score, "stored": False}
    return {"run_id": run_id, "score": body.score, "stored": True}
