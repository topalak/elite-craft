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
        #max_length=1000,
        description="User's question about agent development"
    )


class RetrievedChunk(BaseModel):
    """
    Model for a single retrieved documentation chunk.
    """
    url: str = Field(description="Source URL of the documentation")
    chunk_id_in_document: int = Field(description="Chunk position in document")
    content: str = Field(description="Text content of the chunk")


class QuestionResponse(BaseModel):
    """
    Response model for ask endpoint.

    Returned by: POST /api/ask
    """
    answer: str = Field(description="LLM-generated answer based on retrieved docs")
    retrieved_chunks: list[RetrievedChunk] = Field(
        description="Documentation chunks used to generate answer"
    )

class UpdateDBRequest(BaseModel):
    """
    Request model for database update endpoint.

    Used by: POST /api/update-db
    """
    urls: list[str] = Field(
        ...,
        min_length=1,
        description="List of URLs to crawl and add to database"
    )


class UpdateDBResponse(BaseModel):
    """
    Response model for database update endpoint.

    Returned by: POST /api/update-db
    """
    status: str = Field(description="Processing status")
    message: str = Field(description="Status message")
    urls_count: int = Field(description="Number of URLs being processed")