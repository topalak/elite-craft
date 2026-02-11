"""
FastAPI application for Elite Craft agent API.

This is the backend HTTP server that:
- Exposes REST endpoints for the Crafter agent
- Handles requests from Streamlit frontend
- Returns JSON responses
- Runs on a port you decide
"""
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from elite_craft.agent.crafter_agent import Crafter
from elite_craft.api.schemas import (
    QuestionRequest,
    QuestionResponse,
)
from elite_craft.enums import Provider


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOGGING_LEVEL)

app = FastAPI(
    title="Elite Craft API",
    description="AI Agent API for helping developers build agents",
    version="0.1.0"
)

# Add CORS middleware (allows Streamlit to call this API)
# SECURITY: ALLOWED_ORIGINS should be restricted to your Streamlit domain in production
# Set DEBUG=false and configure ALLOWED_ORIGINS in .env for production deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if not settings.DEBUG else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Crafter agent ONCE when server starts (not per request!)
crafter = Crafter(
    llm_model=settings.LLM_NAME,
    llm_api_key=settings.OLLAMA_API_KEY.get_secret_value(),
    llm_provider=Provider(settings.LLM_PROVIDER),
    ollama_provider_url=settings.OLLAMA_HOST_COLAB.get_secret_value(),
    supabase_url=settings.SUPABASE_URL.get_secret_value(),
    supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY.get_secret_value(),
    embedding_model=settings.EMBEDDING_MODEL,
    tavily_api_key=settings.TAVILY_API_KEY.get_secret_value(),
)

logger.info("✅ Crafter agent initialized")


# ============================================
# ROUTES / ENDPOINTS
# ============================================

@app.post("/api/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """
    Main endpoint: Ask the Crafter agent a question.

    This is the core functionality - users ask questions about
    building agents, and get answers based on retrieved documentation.

    Args:
        request: Contains the user's query

    Returns:
        Answer from LLM and retrieved documentation chunks

    Example:
        POST http://localhost:8000/api/ask
        Body: {"query": "How do I build an agent?"}

    Raises:
        HTTPException: If agent processing fails
    """
    # Input validation
    if not request.query or not request.query.strip():
        logger.warning("Empty query received")
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    try:
        # Call Crafter agent using the ask method
        answer = crafter.ask(query=request.query, print_to_cli=False)

        return QuestionResponse(answer=answer)

    except TimeoutError as e:
        logger.error(f"LLM request timeout: {e}")
        raise HTTPException(
            status_code=504,  # Gateway Timeout
            detail="AI service request timed out. Please try again."
        )

    except ConnectionError as e:
        logger.error(f"LLM connection failed: {e}")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Cannot reach AI service. Please try again later."
        )

    except Exception as e:
        # Unknown error - log with full context
        logger.exception(f"Unexpected error in ask_question: {e}")
        # In production, return generic error message
        if settings.DEBUG:
            detail = f"Error: {str(e)}"
        else:
            detail = "An internal error occurred. Please contact support if this persists."
        raise HTTPException(status_code=500, detail=detail)


# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)