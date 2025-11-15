from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="token-usage-tracker",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Token usage tracking for AI/LLM applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/token-usage-tracker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "starlette>=0.20.0",
        "loguru>=0.7.0",
        "sqlalchemy>=2.0.0",
        "dataclasses-json>=0.5.0",
    ],
    extras_require={
        "otel": [
            "opentelemetry-api>=1.20.0",
            "opentelemetry-sdk>=1.20.0",
            "opentelemetry-instrumentation>=0.41b0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)