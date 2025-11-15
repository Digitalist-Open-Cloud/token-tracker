import time
import json
import uuid
from typing import Optional, Dict, Any
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import ASGIApp
import asyncio

class TokenUsageMiddleware(BaseHTTPMiddleware):
    """ASGI middleware for token tracking with streaming support"""

    def __init__(self, app: ASGIApp, config=None):
        super().__init__(app)
        print("=" * 80)
        print("TokenUsageMiddleware.__init__ called")

        # Import here to avoid circular imports
        from .config import TokenTrackerConfig
        from .logger import get_token_logger

        self.config = config or TokenTrackerConfig.from_env()
        self.logger = get_token_logger(self.config)

        print(f"Config enabled: {self.config.enabled}")
        print(f"File logging enabled: {self.config.file_logging_enabled}")
        print(f"Log file path: {self.config.log_file_path}")
        print(f"OTEL endpoint: {self.config.otel_endpoint}")
        print(f"Monitored endpoints: {self.config.monitored_endpoints}")
        print("=" * 80)

        # Test write directly
        if self.config.enabled:
            self._test_logger()

    def _test_logger(self):
        """Test the logger directly"""
        print("Testing logger directly...")
        try:
            entry_id = self.logger.log_token_usage(
                model="test-model",
                prompt_tokens=10,
                completion_tokens=20,
                endpoint="/test",
                metadata={"test": "init"}
            )
            print(f"Test log entry created: {entry_id}")
        except Exception as e:
            print(f"Test log failed: {e}")
            import traceback
            traceback.print_exc()

    async def dispatch(self, request: Request, call_next):
        """Process the request and track token usage"""
        path = str(request.url.path)
        method = request.method

        print(f"\n>>> Middleware dispatch: {method} {path}")

        # Check if we should track this request
        should_track = self._should_track(request)
        print(f"    Should track: {should_track} (enabled={self.config.enabled})")

        if not self.config.enabled or not should_track:
            print(f"    Skipping tracking")
            return await call_next(request)

        print(f"    TRACKING REQUEST!")

        # Generate request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Store request body
        request_body = None
        if method == "POST":
            try:
                # Read the body
                body_bytes = await request.body()
                request_body = body_bytes.decode("utf-8")
                print(f"    Captured request body: {len(body_bytes)} bytes")

                # Parse to check what we got
                try:
                    req_data = json.loads(request_body)
                    print(f"    Model: {req_data.get('model', 'unknown')}")
                    print(f"    Stream: {req_data.get('stream', False)}")
                    print(f"    Messages: {len(req_data.get('messages', []))}")
                except:
                    pass

                # Create new receive that returns the buffered body
                async def receive():
                    return {
                        "type": "http.request",
                        "body": body_bytes,
                        "more_body": False,
                    }

                request._receive = receive

            except Exception as e:
                print(f"    ERROR capturing request: {e}")
                import traceback
                traceback.print_exc()
                return await call_next(request)

        # Call the actual endpoint
        print(f"    Calling next handler...")
        response = await call_next(request)

        # Get the actual class name
        response_class = response.__class__.__name__
        print(f"    Response type: {response_class}")

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        print(f"    Duration: {duration_ms:.2f}ms")

        # Get user info
        user_id = None
        user_email = None
        if hasattr(request.state, "user"):
            user = request.state.user
            if user:
                user_id = getattr(user, "id", None)
                user_email = getattr(user, "email", None)
                print(f"    User: {user_email} (ID: {user_id})")

        # Check if it's a streaming response (including wrapped ones)
        is_streaming = (
            isinstance(response, StreamingResponse) or
            response_class == "_StreamingResponse" or
            hasattr(response, 'body_iterator')
        )

        if is_streaming:
            print(f"    Processing STREAMING response...")

            # Store original iterator
            original_iterator = response.body_iterator
            chunks = []

            async def capture_and_forward():
                print(f"    Starting to capture chunks...")
                chunk_count = 0
                try:
                    async for chunk in original_iterator:
                        chunk_count += 1
                        chunks.append(chunk)
                        yield chunk

                        # Log every 10 chunks to see progress
                        if chunk_count % 10 == 0:
                            print(f"    ... {chunk_count} chunks captured")

                    print(f"    Total chunks captured: {chunk_count}")

                    # Process after streaming completes
                    if request_body and chunks:
                        try:
                            full_response = b"".join(chunks).decode("utf-8", errors="ignore")
                            print(f"    Full response size: {len(full_response)} bytes")

                            # Show first chunk for debugging
                            if full_response:
                                first_line = full_response.split('\n')[0][:100]
                                print(f"    First line: {first_line}")

                            # Log the token usage
                            self._log_usage(
                                request_body=request_body,
                                response_body=full_response,
                                endpoint=path,
                                user_id=user_id,
                                user_email=user_email,
                                duration_ms=duration_ms,
                                request_id=request_id
                            )
                        except Exception as e:
                            print(f"    ERROR processing captured response: {e}")
                            import traceback
                            traceback.print_exc()

                except Exception as e:
                    print(f"    ERROR in streaming: {e}")
                    import traceback
                    traceback.print_exc()
                    raise  # Re-raise to not break the stream

            # Replace the body iterator
            response.body_iterator = capture_and_forward()

        else:
            print(f"    Processing REGULAR response...")

            # For regular responses, try to get the body
            try:
                if hasattr(response, 'body'):
                    response_body = response.body
                    if response_body:
                        response_text = response_body.decode("utf-8", errors="ignore")
                        print(f"    Response body size: {len(response_text)} bytes")

                        if request_body:
                            self._log_usage(
                                request_body=request_body,
                                response_body=response_text,
                                endpoint=path,
                                user_id=user_id,
                                user_email=user_email,
                                duration_ms=duration_ms,
                                request_id=request_id
                            )
            except Exception as e:
                print(f"    ERROR processing response: {e}")
                import traceback
                traceback.print_exc()

        return response

    def _should_track(self, request: Request) -> bool:
        """Check if this request should be tracked"""
        if request.method != "POST":
            return False

        path = str(request.url.path).lower()

        # Check excluded first
        for excluded in self.config.excluded_endpoints:
            if path.startswith(excluded.lower()):
                print(f"    Path excluded: {path} matches {excluded}")
                return False

        # Check monitored
        for monitored in self.config.monitored_endpoints:
            if path.startswith(monitored.lower()):
                print(f"    Path monitored: {path} matches {monitored}")
                return True

        print(f"    Path not in monitored list: {path}")
        return False

    def _log_usage(
        self,
        request_body: str,
        response_body: str,
        endpoint: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        duration_ms: Optional[float] = None,
        request_id: Optional[str] = None
    ):
        """Log token usage"""
        print(f"\n    === LOGGING TOKEN USAGE ===")

        try:
            # Parse request
            req_data = json.loads(request_body)
            model = req_data.get("model", "unknown")
            streaming = req_data.get("stream", False)

            print(f"    Model: {model}")
            print(f"    Streaming: {streaming}")
            print(f"    Response body preview: {response_body[:200]}")

            # Extract token counts from response
            token_counts = self._extract_tokens(response_body, streaming)
            print(f"    Extracted tokens: {token_counts}")

            # Get completion text
            completion_text = self._extract_completion(response_body, streaming)
            print(f"    Completion text length: {len(completion_text)}")
            if completion_text:
                print(f"    Completion preview: {completion_text[:100]}")

            # Get prompt text (last user message)
            prompt_text = ""
            messages = req_data.get("messages", [])
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        prompt_text = content
                        break
            print(f"    Prompt text length: {len(prompt_text)}")

            # Log via the logger
            print(f"    Calling logger.log_token_usage()...")

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
                }
            )

            print(f"    Entry ID: {entry_id}")
            print(f"    === LOGGING COMPLETE ===\n")

            # Force flush if file logging
            if self.config.file_logging_enabled:
                self.logger.flush()
                print(f"    Flushed to file")

        except Exception as e:
            print(f"    ERROR in _log_usage: {e}")
            import traceback
            traceback.print_exc()

    def _extract_tokens(self, response_body: str, streaming: bool) -> Dict:
        """Extract token usage from response"""
        try:
            if not streaming:
                data = json.loads(response_body)
                if "usage" in data:
                    return {
                        "prompt_tokens": data["usage"].get("prompt_tokens"),
                        "completion_tokens": data["usage"].get("completion_tokens")
                    }
            else:
                # For streaming, look in the chunks
                lines = response_body.split('\n')
                for line in reversed(lines):
                    if line.startswith("data: "):
                        content = line[6:]
                        if content and content != "[DONE]":
                            try:
                                chunk = json.loads(content)
                                if "usage" in chunk:
                                    return {
                                        "prompt_tokens": chunk["usage"].get("prompt_tokens"),
                                        "completion_tokens": chunk["usage"].get("completion_tokens")
                                    }
                            except:
                                continue
        except Exception as e:
            print(f"    Error extracting tokens: {e}")

        return {}

    def _extract_completion(self, response_body: str, streaming: bool) -> str:
        """Extract completion text"""
        try:
            if not streaming:
                data = json.loads(response_body)
                if "choices" in data and data["choices"]:
                    return data["choices"][0].get("message", {}).get("content", "")
            else:
                # For streaming, concatenate chunks
                parts = []
                lines = response_body.split('\n')
                for line in lines:
                    if line.startswith("data: "):
                        content = line[6:]
                        if content and content != "[DONE]":
                            try:
                                chunk = json.loads(content)
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        parts.append(delta["content"])
                            except:
                                continue
                return "".join(parts)
        except Exception as e:
            print(f"    Error extracting completion: {e}")

        return ""