"""
Pydantic models for API request/response validation.

These models define:
- What data the API expects to receive (Request models)
- What data the API returns (Response models)
- Automatic validation by FastAPI
"""
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """
    Request model for asking questions to the agent.

    Used by: POST /api/ask
    """
    query: str = Field(
        ...,
        min_length=1,
        description="User's question about agent development"
    )


class QuestionResponse(BaseModel):
    """
    Response model for ask endpoint.

    Returned by: POST /api/ask
    """
    answer: str = Field(description="LLM-generated answer")