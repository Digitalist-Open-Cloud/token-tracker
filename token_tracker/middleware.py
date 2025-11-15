import time
import json
import uuid
from typing import Optional, Callable, Any, Dict

from .logger import TokenUsageLogger
from .config import TokenTrackerConfig

class TokenUsageMiddleware:
    """Generic middleware for token tracking"""

    def __init__(self, config: Optional[TokenTrackerConfig] = None):
        self.config = config or TokenTrackerConfig.from_env()
        self.logger = TokenUsageLogger(self.config)

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
                self.logger.log_token_usage(
                    model=model,
                    prompt_tokens=token_data["prompt_tokens"],
                    completion_tokens=token_data["completion_tokens"],
                    user_id=user_id,
                    endpoint=endpoint,
                    duration_ms=duration_ms,
                    metadata={
                        "message_count": len(req_data.get("messages", [])),
                        "streaming": req_data.get("stream", False),
                    }
                )

                return token_data

        except Exception as e:
            logger.error(f"Failed to process token usage: {e}")

        return None

    def _extract_token_usage(self, response_body: str) -> Optional[Dict]:
        """Extract token usage from response"""
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
                break

        # Extract response
        completion_text = ""
        try:
            resp_data = json.loads(response_body)
            if "choices" in resp_data:
                completion_text = resp_data["choices"][0].get("message", {}).get("content", "")
        except:
            pass

        return {
            "prompt_tokens": self.logger.estimate_tokens(prompt_text),
            "completion_tokens": self.logger.estimate_tokens(completion_text),
        }