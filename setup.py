from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="token-tracker",
    version="0.1.0",
    author="Your Name",
    description="Token usage tracking with OpenTelemetry for AI/LLM applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "loguru>=0.7.0",
        "python-dateutil>=2.8.0",

        # OpenTelemetry is required
        "opentelemetry-api>=1.37.0",
        "opentelemetry-sdk>=1.37.0",
        "opentelemetry-exporter-otlp>=1.37.0",

        # Web framework support
        "starlette>=0.20.0",
    ],
    extras_require={
        "tiktoken": [
            "tiktoken>=0.5.0",  # For accurate token counting
        ],
        "fastapi": [
            "fastapi>=0.100.0",  # For FastAPI integration
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
