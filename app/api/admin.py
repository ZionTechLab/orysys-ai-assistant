"""Admin-only operational endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import faults
from app.errors import InvalidRequestError, PermissionDeniedError
from app.security.auth import Role, User, get_current_user

router = APIRouter(prefix="/admin", tags=["admin"])


async def admin_user(user: User = Depends(get_current_user)) -> User:
    if user.role != Role.ADMIN:
        raise PermissionDeniedError("Administrators only.")
    return user


class FaultChanges(BaseModel):
    pinecone: bool | None = None
    llm: bool | None = None
    mcp: bool | None = None


@router.get("/faults")
async def get_faults(_: User = Depends(admin_user)) -> dict:
    return faults.current()


@router.post("/faults")
async def set_faults(body: FaultChanges, _: User = Depends(admin_user)) -> dict:
    """Simulate dependency outages to demonstrate graceful degradation (off by default)."""
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        return faults.set_faults(changes)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from None
