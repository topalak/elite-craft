"""
FastAPI backend for Elite Craft agent.

This module provides:
- REST API endpoints for the Crafter agent
- Request/response schemas with validation
- FastAPI application instance
"""

#from elite_craft.api.fastapi_server import app
from elite_craft.api.client import EliteCraftClient


__all__ = [
    # Client
    "EliteCraftClient",

]