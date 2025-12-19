"""
FastAPI application for Elite Craft agent API.

This is the backend HTTP server that:
- Exposes REST endpoints for the Crafter agent
- Handles requests from Streamlit frontend
- Returns JSON responses
- Runs on a port you decide
"""
import logging

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from elite_craft.agent.crafter_agent import Crafter
from elite_craft.services.update_db_pipeline import UpdateDBPipeline
from elite_craft.api.schemas import (
    QuestionRequest,
    QuestionResponse,
    UpdateDBRequest,
    UpdateDBResponse,
)


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOGGING_LEVEL)

# Create FastAPI app
app = FastAPI(
    title="Elite Craft API",
    description="AI Agent API for helping developers build agents",
    version="0.1.0"
)

# Add CORS middleware (allows Streamlit to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Streamlit domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Crafter agent ONCE when server starts (not per request!)
crafter = Crafter(
    llm_model=settings.LLM_NAME,
    llm_api_key=settings.OLLAMA_API_KEY,
    use_ollama_local=settings.USE_OLLAMA_LOCAL,
    ollama_provider_url=settings.OLLAMA_HOST_COLAB,
    supabase_url=settings.SUPABASE_URL,
    supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY,
    embedding_model=settings.EMBEDDING_MODEL,

)

logger.info("✅ Crafter agent initialized")

# Initialize UpdateDBPipeline
pipeline = UpdateDBPipeline(
    embedding_model=settings.EMBEDDING_MODEL,
    supabase_url=settings.SUPABASE_URL,
    supabase_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
)

logger.info("✅ UpdateDBPipeline initialized")


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
        # Call Crafter agent
        result = crafter.ask(request.query)

        return QuestionResponse(
            answer=result,
        )

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
        # Check if it's a database-related error
        error_name = type(e).__name__
        error_msg = str(e)

        if "supabase" in error_name.lower() or "postgrest" in error_name.lower():
            logger.error(f"Database error: {error_msg}")
            # Surface actual error in development for debugging
            raise HTTPException(
                status_code=503,
                detail=f"Database error: {error_msg}"
            )

        # Unknown error - log with full context and surface the actual error
        logger.exception(f"Unexpected error in ask_question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error: {error_msg}"
        )


@app.post("/api/update-db", response_model=UpdateDBResponse)
async def update_database(
    request: UpdateDBRequest,
    background_tasks: BackgroundTasks
) -> UpdateDBResponse:
    """
    Trigger database update with new documentation URLs.

    This endpoint crawls the provided URLs, chunks the content,
    generates embeddings, and stores them in Supabase.
    The process runs in the background to avoid blocking the request.

    Args:
        request: Contains list of URLs to process
        background_tasks: FastAPI background task manager

    Returns:
        Status message indicating the update has started

    Example:
        POST http://localhost:8000/api/update-db
        Body: {"urls": ["https://docs.langchain.com/..."]}
    """
    # Input validation
    if not request.urls:
        raise HTTPException(
            status_code=400,
            detail="URL list cannot be empty"
        )

    try:
        logger.info(f"Database update requested for {len(request.urls)} URLs")

        # Add task to background - doesn't block the response
        background_tasks.add_task(
            pipeline.process_multiple_urls,
            request.urls
        )

        return UpdateDBResponse(
            status="started",
            message=f"Database update started for {len(request.urls)} URLs",
            urls_count=len(request.urls)
        )

    except TimeoutError as e:
        logger.error(f"Database connection timeout: {e}")
        raise HTTPException(
            status_code=504,  # Gateway Timeout
            detail="Database connection timed out. Please try again."
        )

    except ConnectionError as e:
        logger.error(f"Cannot connect to database: {e}")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Database connection unavailable"
        )

    except Exception as e:
        logger.exception(f"Unexpected error starting database update: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start database update"
        )


# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)