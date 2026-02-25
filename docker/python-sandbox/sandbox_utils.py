"""
Sandbox utilities for code execution.

This module provides pre-configured objects for use in the sandbox environment.
The 'model' object is created automatically when you import this module.

Example:
    from sandbox_utils import model
    from langchain.agents import create_agent

    agent = create_agent(
        model=model,
        tools=[...],
        system_prompt="..."
    )
"""

import os

from config import settings

from langchain_ollama import ChatOllama

# Pre-configured model for sandbox LLM access
# This is created when the module is imported
# The model uses the proxy server for secure API access

#todo we might need to add the trace pattern here
_proxy_secret = os.environ.get("PROXY_SECRET")
if not _proxy_secret:
    raise RuntimeError(
        "PROXY_SECRET environment variable is required for sandbox LLM access. "
        "Ensure the code executor injects it when starting the container."
    )

model = ChatOllama(
    model=settings.SANDBOX_LLM_NAME,
    base_url="http://host.docker.internal:4000",
    client_kwargs={
        'headers': {'Authorization': f'Bearer {_proxy_secret}'}
    },
    temperature=0
)

__all__ = ["model"]