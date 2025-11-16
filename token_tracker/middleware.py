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

        print(f"[TokenTracker] Initialized - Enabled: {self.config.enabled}")
        print(f"[TokenTracker] File logging: {self.config.file_logging_enabled}")

    async def dispatch(self, request: Request, call_next):
        """Process the request and track token usage"""
        path = str(request.url.path)
        method = request.method

        # Check if we should track this request
        if not self.config.enabled or not self._should_track(request):
            return await call_next(request)

        print(f"\n[TokenTracker] Tracking: {method} {path}")

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

                print(f"[TokenTracker] Model: {req_data.get('model', 'unknown')}")
                print(f"[TokenTracker] Stream: {req_data.get('stream', False)}")

                # Special logging for /api/chat/completed
                if path == "/api/chat/completed":
                    messages = req_data.get('messages', [])
                    print(f"[TokenTracker] /api/chat/completed - {len(messages)} messages")
                    # The last assistant message contains the completion
                    for msg in reversed(messages):
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            print(f"[TokenTracker] Assistant response length: {len(content)} chars")
                            break

                # Create new receive that returns the buffered body
                async def receive():
                    return {
                        "type": "http.request",
                        "body": body_bytes,
                        "more_body": False,
                    }

                request._receive = receive

            except Exception as e:
                print(f"[TokenTracker] ERROR capturing request: {e}")
                return await call_next(request)

        # Call the actual endpoint
        response = await call_next(request)

        # Extract user info from request.state._state
        user_id = None
        user_email = None

        if hasattr(request, 'state') and hasattr(request.state, '_state'):
            state_data = request.state._state

            # Get metadata from state
            metadata = state_data.get('metadata', {})
            if metadata:
                user_id = metadata.get('user_id')
                # Get user name from variables
                variables = metadata.get('variables', {})
                user_name = variables.get('{{USER_NAME}}', '')

                # For now, use username as email placeholder
                user_email = user_name if user_name and user_name != 'Admin' else None

                print(f"[TokenTracker] User extracted - ID: {user_id}, Name: {user_name}")

        # Special handling for /api/chat/completed endpoint
        if path == "/api/chat/completed":
            print(f"[TokenTracker] Processing /api/chat/completed")

            # Get the completion from the request (not response)
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
                print(f"[TokenTracker] Found completion to log: {len(assistant_response)} chars")

                # Check the response body for any token usage info
                if hasattr(response, 'body'):
                    try:
                        response_text = response.body.decode("utf-8", errors="ignore")
                        print(f"[TokenTracker] /api/chat/completed response: {response_text[:500]}")

                        # Try to parse response for token info
                        try:
                            resp_data = json.loads(response_text)
                            # Check if there's usage info in the response
                            if 'usage' in resp_data:
                                print(f"[TokenTracker] Found usage in response: {resp_data['usage']}")
                        except:
                            pass
                    except:
                        response_text = ""
                else:
                    response_text = ""

                # Log the token usage
                entry_id = self.logger.log_token_usage(
                    model=req_data.get('model', 'unknown'),
                    prompt_text=user_message,
                    completion_text=assistant_response,
                    user_id=user_id,
                    user_email=user_email,
                    endpoint=path,
                    request_id=request_id,
                    duration_ms=(time.time() - start_time) * 1000,
                    streaming=False,  # This endpoint is after streaming completes
                    temperature=req_data.get('temperature'),
                    max_tokens=req_data.get('max_tokens'),
                    metadata={
                        "message_count": len(messages),
                        "endpoint_type": "completed"
                    }
                )

                if entry_id:
                    print(f"[TokenTracker] Successfully logged completion: {entry_id}")

                # Force flush
                if self.config.file_logging_enabled:
                    self.logger.flush()

            return response

        # Skip the task creation response from /api/chat/completions
        if path == "/api/chat/completions" and hasattr(response, 'body'):
            try:
                body_text = response.body.decode("utf-8", errors="ignore")
                if '"task_id"' in body_text and len(body_text) < 200:
                    print(f"[TokenTracker] Skipping task creation response")
                    return response
            except:
                pass

        # Check if it's a streaming response (for other endpoints)
        is_streaming_response = hasattr(response, 'body_iterator') and req_data.get('stream', False)

        if is_streaming_response:
            print(f"[TokenTracker] Handling STREAMING response")

            # ... (rest of streaming handler code stays the same)
            # But we probably won't need it since real completion is in /api/chat/completed

        else:
            # Non-streaming response
            print(f"[TokenTracker] Handling NON-STREAMING response")

            # Only process non-/api/chat/completed endpoints here
            if path != "/api/chat/completed":
                try:
                    if hasattr(response, 'body'):
                        response_body = response.body
                        if response_body and req_data:
                            response_text = response_body.decode("utf-8", errors="ignore")

                            await self._process_and_log(
                                response_body=response_text,
                                req_data=req_data,
                                endpoint=path,
                                user_id=user_id,
                                user_email=user_email,
                                duration_ms=(time.time() - start_time) * 1000,
                                request_id=request_id
                            )
                except Exception as e:
                    print(f"[TokenTracker] Error processing response: {e}")
                    import traceback
                    traceback.print_exc()

        return response

    async def _process_buffered_response(
        self,
        chunks_buffer: List[bytes],
        req_data: dict,
        endpoint: str,
        user_id: Optional[str],
        user_email: Optional[str],
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
                user_email=user_email,
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
        user_email: Optional[str],
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
                user_email=user_email,
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