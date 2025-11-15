import time
import json
import uuid
from typing import Optional, Callable, Any, Dict, MutableMapping, cast
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.datastructures import Headers
import asyncio

from .logger import TokenUsageLogger
from .config import TokenTrackerConfig


class TokenUsageMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for token tracking"""

    def __init__(self, app, config: Optional[TokenTrackerConfig] = None):
        super().__init__(app)
        self.config = config or TokenTrackerConfig.from_env()
        self.logger = TokenUsageLogger(self.config)

        # Log initialization
        if self.config.enabled:
            print(f"TokenUsageMiddleware initialized - tracking enabled")
        else:
            print(f"TokenUsageMiddleware initialized - tracking disabled")

    async def dispatch(self, request: Request, call_next):
        """Process the request and track token usage"""

        # Skip if not enabled or not a monitored endpoint
        if not self.config.enabled or not self._should_track(request):
            return await call_next(request)

        # Capture start time
        start_time = time.time()

        # Capture request body
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body_bytes = await request.body()
                request_body = body_bytes.decode("utf-8")
                # Need to reconstruct the request for downstream
                from starlette.datastructures import State

                async def receive():
                    return {"type": "http.request", "body": body_bytes}

                request._receive = receive
            except Exception as e:
                print(f"Failed to capture request body: {e}")

        # Process the request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # For streaming responses, we can't easily capture the body
        # For now, log what we can
        if request_body:
            # Get user info if available
            user_id = None
            if hasattr(request.state, "user"):
                user = request.state.user
                user_id = getattr(user, "id", None) or getattr(user, "user_id", None)

            # Process and log
            try:
                self.process_request(
                    request_body=request_body,
                    response_body="",  # Would need more complex handling for response
                    endpoint=str(request.url.path),
                    user_id=user_id,
                    duration_ms=duration_ms
                )
            except Exception as e:
                print(f"Failed to process token usage: {e}")

        return response

    def _should_track(self, request: Request) -> bool:
        """Check if this request should be tracked"""
        path = str(request.url.path).lower()

        # Check excluded endpoints first
        for excluded in self.config.excluded_endpoints:
            if path.startswith(excluded.lower()):
                return False

        # Check monitored endpoints
        if self.config.monitored_endpoints:
            for monitored in self.config.monitored_endpoints:
                if path.startswith(monitored.lower()):
                    return True
            return False  # If monitor list exists, only track those

        # Default: track all non-excluded endpoints
        return True

    def process_request(
        self,
        request_body: str,
        response_body: str,
        endpoint: str,
        user_id: Optional[str] = None,
        duration_ms: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Process a request/response pair and log token usage"""

        if not self.config.enabled:
            return None

        try:
            # Parse request
            req_data = json.loads(request_body) if request_body else {}
            model = req_data.get("model", "unknown")

            # Extract or estimate tokens
            token_data = self._extract_token_usage(response_body)

            if not token_data and self.config.estimate_tokens:
                token_data = self._estimate_tokens(req_data, response_body)

            if token_data:
                # Extract additional info from request
                messages = req_data.get("messages", [])

                # Get current question (last user message)
                current_question = ""
                for msg in reversed(messages):
                    if msg.get("role") == "user":
                        current_question = msg.get("content", "")
                        break

                self.logger.log_token_usage(
                    model=model,
                    prompt_tokens=token_data["prompt_tokens"],
                    completion_tokens=token_data["completion_tokens"],
                    prompt_text=current_question if self.config.store_samples else None,
                    completion_text=response_body[:500] if self.config.store_samples else None,
                    user_id=user_id,
                    endpoint=endpoint,
                    duration_ms=duration_ms,
                    streaming=req_data.get("stream", False),
                    temperature=req_data.get("temperature"),
                    max_tokens=req_data.get("max_tokens"),
                    metadata={
                        "message_count": len(messages),
                        "streaming": req_data.get("stream", False),
                    }
                )

                return token_data

        except Exception as e:
            print(f"Failed to process token usage: {e}")

        return None

    def _extract_token_usage(self, response_body: str) -> Optional[Dict]:
        """Extract token usage from response"""
        if not response_body:
            return None

        try:
            resp_data = json.loads(response_body)
            if "usage" in resp_data:
                return {
                    "prompt_tokens": resp_data["usage"].get("prompt_tokens", 0),
                    "completion_tokens": resp_data["usage"].get("completion_tokens", 0),
                }
        except:
            pass
        return None

    def _estimate_tokens(self, request_data: dict, response_body: str) -> Dict:
        """Estimate tokens when not provided"""
        # Extract last user message
        messages = request_data.get("messages", [])
        prompt_text = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                prompt_text = msg.get("content", "")
                if isinstance(prompt_text, str):
                    break
                prompt_text = ""

        # Extract response
        completion_text = ""
        if response_body:
            try:
                resp_data = json.loads(response_body)
                if "choices" in resp_data and resp_data["choices"]:
                    message = resp_data["choices"][0].get("message", {})
                    completion_text = message.get("content", "")
            except:
                pass

        # Use the logger's token counter
        prompt_tokens = 0
        completion_tokens = 0

        if hasattr(self.logger, 'token_counter'):
            prompt_tokens, _ = self.logger.token_counter.count_tokens(prompt_text, request_data.get("model"))
            completion_tokens, _ = self.logger.token_counter.count_tokens(completion_text, request_data.get("model"))
        else:
            # Fallback to simple estimation
            prompt_tokens = max(1, len(prompt_text) // 4)
            completion_tokens = max(1, len(completion_text) // 4)

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
