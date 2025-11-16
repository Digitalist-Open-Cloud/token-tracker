import time
import json
import uuid
from typing import Optional, Dict, Any, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import ASGIApp
import asyncio

class TokenUsageMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for token tracking with streaming support"""

    def __init__(self, app: ASGIApp, config=None):
        super().__init__(app)

        # Import here to avoid circular imports
        from .config import TokenTrackerConfig
        from .logger import get_token_logger

        self.config = config or TokenTrackerConfig.from_env()
        self.logger = get_token_logger(self.config)

        # Cache to store user info between related requests
        self.request_cache = {}

        print(f"[TokenTracker] Initialized - Enabled: {self.config.enabled}")
        print(f"[TokenTracker] File logging: {self.config.file_logging_enabled}")

    async def dispatch(self, request: Request, call_next):
        """Process the request and track token usage"""
        path = str(request.url.path)
        method = request.method

        # Check if we should track this request
        if not self.config.enabled or not self._should_track(request):
            return await call_next(request)

        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Store request body and parsed data
        request_body = None
        req_data = {}
        if method == "POST":
            try:
                body_bytes = await request.body()
                request_body = body_bytes.decode("utf-8")
                req_data = json.loads(request_body)

                # Create new receive that returns the buffered body
                async def receive():
                    return {
                        "type": "http.request",
                        "body": body_bytes,
                        "more_body": False,
                    }

                request._receive = receive  # noqa: SLF001

            except Exception as e:
                print(f"[TokenTracker] ERROR capturing request: {e}")
                return await call_next(request)

        # Call the actual endpoint
        response = await call_next(request)

        # Extract user info from request.state._state (for /api/chat/completions)
        if path == "/api/chat/completions":
            if hasattr(request, 'state') and hasattr(request.state, '_state'):
                state_data = request.state._state  # noqa: SLF001

                # Get metadata from state
                metadata = state_data.get('metadata', {})
                if metadata:
                    user_id = metadata.get('user_id')
                    chat_id = metadata.get('chat_id')
                    session_id = metadata.get('session_id')
                    message_id = metadata.get('message_id')

                    # Get variables
                    variables = metadata.get('variables', {})
                    user_name = variables.get('{{USER_NAME}}', '')

                    # Cache this info
                    cache_key = f"{req_data.get('model', 'unknown')}"
                    self.request_cache[cache_key] = {
                        'user_id': user_id,
                        'user_name': user_name,
                        'chat_id': chat_id,
                        'session_id': session_id,
                        'message_id': message_id,
                        'timestamp': time.time()
                    }

                    # Clean old cache entries
                    current_time = time.time()
                    keys_to_delete = [
                        k for k, v in self.request_cache.items()
                        if current_time - v.get('timestamp', 0) > 60
                    ]
                    for k in keys_to_delete:
                        del self.request_cache[k]

        # Special handling for /api/chat/completed endpoint
        elif path == "/api/chat/completed":
            # Try to get user info from cache
            cache_key = f"{req_data.get('model', 'unknown')}"
            cached_info = self.request_cache.get(cache_key, {})

            user_id = cached_info.get('user_id')
            user_name = cached_info.get('user_name')
            chat_id = cached_info.get('chat_id')
            session_id = cached_info.get('session_id')
            message_id = cached_info.get('message_id')

            # Get the completion from the request
            messages = req_data.get('messages', [])

            # Extract user message and assistant response
            user_message = ""
            assistant_response = ""

            for msg in messages:
                if msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                elif msg.get('role') == 'assistant':
                    assistant_response = msg.get('content', '')

            if assistant_response:
                # Log the token usage
                entry_id = self.logger.log_token_usage(
                    model=req_data.get('model', 'unknown'),
                    prompt_text=user_message,
                    completion_text=assistant_response,
                    user_id=user_id,
                    session_id=session_id,
                    conversation_id=chat_id,
                    endpoint=path,
                    request_id=request_id,
                    duration_ms=(time.time() - start_time) * 1000,
                    streaming=False,
                    temperature=req_data.get('temperature'),
                    max_tokens=req_data.get('max_tokens'),
                    metadata={
                        "message_count": len(messages),
                        "endpoint_type": "completed",
                        "message_id": message_id,
                        "user_name": user_name,
                    }
                )

                if self.config.file_logging_enabled:
                    self.logger.flush()

        return response

    async def _process_buffered_response(
        self,
        chunks_buffer: List[bytes],
        req_data: dict,
        endpoint: str,
        user_id: Optional[str],
        start_time: float,
        request_id: str
    ):
        """Process the buffered streaming response"""
        try:
            # Combine all chunks
            full_response = b"".join(chunks_buffer).decode("utf-8", errors="ignore")
            print(f"[TokenTracker] Processing buffered response: {len(full_response)} bytes")

            # Calculate final duration
            duration_ms = (time.time() - start_time) * 1000

            # Process and log
            await self._process_and_log(
                response_body=full_response,
                req_data=req_data,
                endpoint=endpoint,
                user_id=user_id,
                duration_ms=duration_ms,
                request_id=request_id
            )

        except Exception as e:
            print(f"[TokenTracker] Error processing buffered response: {e}")
            import traceback
            traceback.print_exc()

    async def _process_and_log(
        self,
        response_body: str,
        req_data: dict,
        endpoint: str,
        user_id: Optional[str],
        duration_ms: float,
        request_id: str
    ):
        """Process and log the token usage"""
        try:
            model = req_data.get("model", "unknown")
            streaming = req_data.get("stream", False)
            messages = req_data.get("messages", [])

            # Show what we're processing
            print(f"[TokenTracker] Processing response for model: {model}")
            if response_body:
                print(f"[TokenTracker] Response preview: {response_body[:200]}...")

            # Extract tokens and text
            token_counts = self._extract_tokens(response_body, streaming)
            completion_text = self._extract_completion(response_body, streaming)

            # Get prompt (last user message)
            prompt_text = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        prompt_text = content
                        break

            print(f"[TokenTracker] Extracted - Tokens: {token_counts}, Completion: {len(completion_text)} chars")

            # Log the usage
            entry_id = self.logger.log_token_usage(
                model=model,
                prompt_tokens=token_counts.get("prompt_tokens"),
                completion_tokens=token_counts.get("completion_tokens"),
                prompt_text=prompt_text if not token_counts.get("prompt_tokens") else None,
                completion_text=completion_text if not token_counts.get("completion_tokens") else None,
                user_id=user_id,
                endpoint=endpoint,
                request_id=request_id,
                duration_ms=duration_ms,
                streaming=streaming,
                temperature=req_data.get("temperature"),
                max_tokens=req_data.get("max_tokens"),
                metadata={
                    "message_count": len(messages),
                    "top_p": req_data.get("top_p"),
                    "frequency_penalty": req_data.get("frequency_penalty"),
                    "presence_penalty": req_data.get("presence_penalty"),
                }
            )

            if entry_id:
                print(f"[TokenTracker] Successfully logged with ID: {entry_id}")
            else:
                print(f"[TokenTracker] Failed to create log entry")

            # Force flush if file logging is enabled
            if self.config.file_logging_enabled:
                self.logger.flush()

        except Exception as e:
            print(f"[TokenTracker] Error in _process_and_log: {e}")
            import traceback
            traceback.print_exc()

    def _should_track(self, request: Request) -> bool:
        """Check if this request should be tracked"""
        path = str(request.url.path).lower()

        # Debug: Log all POST requests to find the actual streaming endpoint
        if request.method == "POST":
            print(f"[TokenTracker Debug] POST request to: {path}")

        if request.method != "POST":
            return False

        # Check excluded first
        for excluded in self.config.excluded_endpoints:
            if path.startswith(excluded.lower()):
                return False

        # Check monitored
        for monitored in self.config.monitored_endpoints:
            if path.startswith(monitored.lower()):
                return True

        return False

    def _extract_tokens(self, response_body: str, streaming: bool) -> Dict:
        """Extract token usage from response"""
        try:
            if not streaming:
                # Non-streaming format
                data = json.loads(response_body)
                if "usage" in data:
                    return {
                        "prompt_tokens": data["usage"].get("prompt_tokens"),
                        "completion_tokens": data["usage"].get("completion_tokens")
                    }
            else:
                # Streaming SSE format
                lines = response_body.split('\n')
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith("data: "):  # FIXED: Complete string literal
                        content = line[6:]  # Remove "data: " prefix
                        if content and content != "[DONE]":
                            try:
                                chunk = json.loads(content)
                                # Check for usage data
                                if "usage" in chunk:
                                    return {
                                        "prompt_tokens": chunk["usage"].get("prompt_tokens"),
                                        "completion_tokens": chunk["usage"].get("completion_tokens")
                                    }
                                # Check x_groq format
                                if "x_groq" in chunk and "usage" in chunk["x_groq"]:
                                    usage = chunk["x_groq"]["usage"]
                                    return {
                                        "prompt_tokens": usage.get("prompt_tokens"),
                                        "completion_tokens": usage.get("completion_tokens")
                                    }
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            print(f"[TokenTracker] Token extraction error: {e}")

        return {}

    def _extract_completion(self, response_body: str, streaming: bool) -> str:
        """Extract completion text from response"""
        try:
            if not streaming:
                # Non-streaming format
                data = json.loads(response_body)
                if "choices" in data and data["choices"]:
                    message = data["choices"][0].get("message", {})
                    return message.get("content", "")
            else:
                # Streaming SSE format
                parts = []
                lines = response_body.split('\n')

                for line in lines:
                    line = line.strip()
                    if line.startswith("data: "):  # Complete string literal
                        content = line[6:]  # Remove "data: " prefix
                        if content and content != "[DONE]":
                            try:
                                chunk = json.loads(content)
                                if "choices" in chunk and chunk["choices"]:
                                    choice = chunk["choices"][0]

                                    # Check for delta content (streaming)
                                    if "delta" in choice:
                                        delta = choice["delta"]
                                        if "content" in delta:
                                            parts.append(delta["content"])

                                    # Check for message content
                                    elif "message" in choice:
                                        message = choice["message"]
                                        if "content" in message:
                                            parts.append(message["content"])

                                    # Check for direct text field
                                    elif "text" in choice:
                                        parts.append(choice["text"])

                            except json.JSONDecodeError:
                                continue

                return "".join(parts)

        except Exception as e:
            print(f"[TokenTracker] Completion extraction error: {e}")

        return ""