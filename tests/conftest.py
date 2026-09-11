"""Disable external model tracing and provide inert credentials for import-only tests."""

import os

os.environ.setdefault("OPENAI_API_KEY", "test-not-a-real-key")
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
os.environ.setdefault("LONG_TERM_MEMORY_ENABLED", "false")
