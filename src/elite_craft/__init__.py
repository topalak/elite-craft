"""
Elite Craft - AI Agent Development Assistant.

An agent framework built to help developers create and enhance their agentic AI projects.
Powered by LangChain, LangGraph, and Pydantic with Supabase for knowledge management.
"""

from elite_craft.services import (
    Crawler,
    Chunker,
    Embedder,
    SupabaseUploadService,
    UpdateDBPipeline,
)

__version__ = "0.1.0"

__all__ = [
    "Crawler",
    "Chunker",
    "Embedder",
    "SupabaseUploadService",
    "UpdateDBPipeline",
]