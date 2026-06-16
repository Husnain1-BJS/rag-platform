"""API key authentication via request header."""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from .config import settings


SKIP_AUTH_PATHS = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}


def get_api_key(request: Request) -> str:
    api_key = request.headers.get("x-api-key", "")
    return api_key


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not settings.API_KEY:
            return await call_next(request)

        path = request.url.path
        if path in SKIP_AUTH_PATHS or path.startswith(("/docs", "/openapi", "/redoc")):
            return await call_next(request)

        key = get_api_key(request)
        if key != settings.API_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Provide it via the x-api-key header."},
            )

        return await call_next(request)