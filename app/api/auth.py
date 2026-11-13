"""Login and current-user endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.security.auth import User, authenticate, create_token, get_current_user
from app.security.rate_limit import LOGIN_LIMITER, enforce, login_limit

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    user_id: str
    display_name: str
    department: str
    role: str
    clearance: list[str]

    @classmethod
    def from_user(cls, user: User) -> "UserOut":
        return cls(
            user_id=user.user_id,
            display_name=user.display_name,
            department=user.department,
            role=user.role.value,
            clearance=user.clearance,
        )


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest) -> LoginResponse:
    # Per-username limit before the password check: slows down password guessing.
    await enforce(LOGIN_LIMITER, body.username.lower(), login_limit(), scope="login")
    user = authenticate(body.username, body.password)
    token, expires_in = create_token(user)
    return LoginResponse(access_token=token, expires_in=expires_in, user=UserOut.from_user(user))


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut.from_user(user)
