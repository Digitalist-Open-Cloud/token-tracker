from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time

from .middleware import TokenUsageMiddleware
from .config import TokenTrackerConfig

class FastAPITokenMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette specific middleware"""

    def __init__(self, app, config: Optional[TokenTrackerConfig] = None):
        super().__init__(app)
        self.tracker = TokenUsageMiddleware(config)

    async def dispatch(self, request: Request, call_next):
        # Check if we should track this endpoint
        if not any(request.url.path.startswith(ep)
                  for ep in self.tracker.config.monitored_endpoints):
            return await call_next(request)

        # Capture request body
        request_body = await request.body()

        # Time the request
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Get user ID if available
        user_id = getattr(request.state, "user_id", None)

        # Process and log
        # Note: You'd need to capture response body here
        self.tracker.process_request(
            request_body.decode(),
            "",  # response_body would need to be captured
            str(request.url.path),
            user_id=user_id,
            duration_ms=duration_ms
        )

        return response