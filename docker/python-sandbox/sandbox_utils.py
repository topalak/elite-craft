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
from langchain_ollama import ChatOllama

# Pre-configured model for sandbox LLM access
# This is created when the module is imported
# The model uses the proxy server for secure API access

#todo we might need to add the trace pattern here
model = ChatOllama(
    model="gpt-oss:20b-cloud",
    base_url="http://host.docker.internal:4000",
    client_kwargs={
        'headers': {'Authorization': f'Bearer {os.environ.get("PROXY_SECRET", "")}'}
    },
    temperature=0
)

__all__ = ["model"]