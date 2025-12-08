"""
FastAPI backend for Elite Craft agent.

This module provides:
- REST API endpoints for the Crafter agent
- Request/response schemas with validation
- FastAPI application instance
"""

from elite_craft.api.fastapi_server import app
from elite_craft.api.fastapi_client import EliteCraftClient
from elite_craft.api.schemas import (
    QuestionRequest,
    QuestionResponse,
    RetrievedChunk,
    UpdateDBRequest,
    UpdateDBResponse,
)

__all__ = [
    # FastAPI app
    "app",
    # Client
    "EliteCraftClient",
    # Schemas
    "QuestionRequest",
    "UpdateDBRequest",
    "QuestionResponse",
    "RetrievedChunk",
    "UpdateDBResponse",
]