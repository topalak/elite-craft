"""
Elite Craft - AI Agent Development Assistant.

An agent framework built to help developers create and enhance their agentic AI projects.
Powered by LangChain, LangGraph, and Pydantic with Supabase for knowledge management.
"""

from elite_craft.services import (
    Chunker,
    Embedder,
    SupabaseUploadService,
    UpdateDBPipeline,
)
from elite_craft.services.crawling import crawl

__version__ = "0.1.0"

__all__ = [
    #service classes
    "Chunker",
    "Embedder",
    "SupabaseUploadService",

    #service functions
    "crawl",

    #pipeline
    "UpdateDBPipeline",
]