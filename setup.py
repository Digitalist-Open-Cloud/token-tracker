from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="token-tracker",
    version="0.1.0",
    author="Mikke Schirén",
    description="Token usage tracking with OpenTelemetry for Open WebUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "loguru>=0.7.0",
        "python-dateutil>=2.8.0",
        "opentelemetry-api>=1.37.0",
        "opentelemetry-sdk>=1.37.0",
        "opentelemetry-exporter-otlp>=1.37.0",
        "starlette>=0.20.0",
    ],
    extras_require={
        "tiktoken": [
            "tiktoken>=0.5.0",
        ],
        "fastapi": [
            "fastapi>=0.100.0",
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
