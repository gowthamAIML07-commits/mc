"""Production Security Headers, Correlation Tracking, and DoS Mitigation Middlewares."""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

HTTP_413_CONTENT_TOO_LARGE = 413


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Enforces strict HTTP security headers across all API and static responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)

        # Apply security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'"
        )

        # Enable HSTS in production or when accessed via HTTPS
        if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        return response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Tracks and propagates unique correlation IDs across request lifecycles."""

    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID") or request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Guards against memory exhaustion by enforcing a strict maximum request payload size."""

    def __init__(self, app, max_content_length: int = 20 * 1024 * 1024):  # Default: 20MB
        super().__init__(app)
        self.max_content_length = max_content_length

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self.max_content_length:
                    return JSONResponse(
                        status_code=HTTP_413_CONTENT_TOO_LARGE,
                        content={
                            "detail": f"Request payload size exceeds maximum permitted limit of {self.max_content_length // (1024 * 1024)}MB."
                        }
                    )
            except ValueError:
                pass

        return await call_next(request)
