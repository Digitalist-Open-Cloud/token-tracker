# telemetry.py
from typing import Any, Optional, Dict
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

class TelemetryBackend:
    def __init__(self, config):
        self.config = config
        self.tracer = None
        self.meter = None
        self._init_telemetry()

    def _init_telemetry(self):
        """Initialize OpenTelemetry with proper API usage"""
        try:
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.config.otel_service_name,
                ResourceAttributes.SERVICE_VERSION: "1.0.0",
            })

            # Initialize tracing if endpoint configured
            if self.config.otel_endpoint:
                # Set up tracing
                trace_provider = TracerProvider(resource=resource)
                trace.set_tracer_provider(trace_provider)

                # Create OTLP exporter
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.config.otel_endpoint,
                    insecure=self.config.otel_insecure,
                    headers=self.config.otel_headers or None
                )

                # Add span processor
                span_processor = BatchSpanProcessor(otlp_exporter)
                trace_provider.add_span_processor(span_processor)

                # Get tracer - FIXED: don't pass version as keyword argument
                self.tracer = trace.get_tracer(
                    self.config.otel_service_name,
                    "1.0.0"  # Pass version as positional argument
                )

                # Set up metrics
                metric_reader = PeriodicExportingMetricReader(
                    exporter=OTLPMetricExporter(
                        endpoint=self.config.otel_endpoint,
                        insecure=self.config.otel_insecure,
                        headers=self.config.otel_headers or None
                    ),
                    export_interval_millis=self.config.otel_export_interval * 1000
                )

                meter_provider = MeterProvider(
                    resource=resource,
                    metric_readers=[metric_reader]
                )
                metrics.set_meter_provider(meter_provider)

                # Get meter
                self.meter = metrics.get_meter(
                    self.config.otel_service_name,
                    "1.0.0"
                )

                # Create metrics instruments
                self._create_metrics()

                print(f"TelemetryBackend initialized with endpoint: {self.config.otel_endpoint}")
            else:
                print("TelemetryBackend: No OTEL endpoint configured, using no-op tracer")
                # Use default no-op tracer
                self.tracer = trace.get_tracer(self.config.otel_service_name)
                self.meter = metrics.get_meter(self.config.otel_service_name)

        except Exception as e:
            print(f"TelemetryBackend initialization error: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to no-op
            self.tracer = trace.get_tracer(self.config.otel_service_name)
            self.meter = metrics.get_meter(self.config.otel_service_name)

    def _create_metrics(self):
        """Create metric instruments"""
        if not self.meter:
            return

        try:
            # Token counters
            self.token_counter = self.meter.create_counter(
                name="token_usage_total",
                description="Total tokens used",
                unit="tokens"
            )

            # Cost counter
            self.cost_counter = self.meter.create_counter(
                name="token_usage_cost",
                description="Total cost of token usage",
                unit="USD"
            )

            # Duration histogram
            self.duration_histogram = self.meter.create_histogram(
                name="token_request_duration",
                description="Duration of token requests",
                unit="ms"
            )

        except Exception as e:
            print(f"Failed to create metrics: {e}")

    def record_metrics(self, entry: Any):
        """Record metrics from a token usage entry"""
        if not self.meter:
            return

        try:
            # Common attributes
            attributes = {
                "model": entry.model,
                "provider": entry.provider,
                "endpoint": entry.endpoint or "unknown",
                "streaming": str(entry.streaming).lower(),
                "token_source": entry.token_source,
            }

            if entry.user_id:
                attributes["user_id"] = entry.user_id

            # Record token counts
            if hasattr(self, 'token_counter'):
                self.token_counter.add(
                    entry.prompt_tokens,
                    attributes={**attributes, "token_type": "prompt"}
                )
                self.token_counter.add(
                    entry.completion_tokens,
                    attributes={**attributes, "token_type": "completion"}
                )

            # Record cost
            if hasattr(self, 'cost_counter') and entry.total_cost:
                self.cost_counter.add(entry.total_cost, attributes=attributes)

            # Record duration
            if hasattr(self, 'duration_histogram') and entry.duration_ms:
                self.duration_histogram.record(entry.duration_ms, attributes=attributes)

        except Exception as e:
            print(f"Failed to record metrics: {e}")

    def create_span(self, entry: Any):
        """Create a trace span for token usage"""
        if not self.tracer:
            return

        try:
            with self.tracer.start_as_current_span(
                name=f"token_usage.{entry.model}",
                attributes={
                    "model": entry.model,
                    "provider": entry.provider,
                    "prompt_tokens": entry.prompt_tokens,
                    "completion_tokens": entry.completion_tokens,
                    "total_tokens": entry.total_tokens,
                    "streaming": entry.streaming,
                    "token_source": entry.token_source,
                    "endpoint": entry.endpoint or "",
                    "user_id": entry.user_id or "",
                }
            ) as span:
                # Add events
                span.add_event(
                    "token_usage_recorded",
                    attributes={
                        "tokens": entry.total_tokens,
                        "cost": entry.total_cost or 0
                    }
                )

                # Set status
                if entry.status == "success":
                    span.set_status(trace.Status(trace.StatusCode.OK))
                else:
                    span.set_status(
                        trace.Status(
                            trace.StatusCode.ERROR,
                            entry.error_message or "Unknown error"
                        )
                    )

        except Exception as e:
            print(f"Failed to create span: {e}")

    def close(self):
        """Cleanup resources"""
        try:
            if self.tracer:
                # Force flush any pending spans
                provider = trace.get_tracer_provider()
                if hasattr(provider, 'shutdown'):
                    provider.shutdown()
        except Exception as e:
            print(f"Error closing telemetry: {e}")
