"""
FastAPI application for Elite Craft agent API.

This is the backend HTTP server that:
- Exposes REST endpoints for the Crafter agent
- Handles requests from Streamlit frontend
- Returns JSON responses
- Runs on port 8000
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
    use_ollama_local=settings.USE_OLLAMA_LOCAL,
    ollama_provider_url=settings.OLLAMA_HOST_LOCAL,
    llm_api_key=settings.OLLAMA_API_KEY,
    embedding_model_name=settings.EMBEDDING_MODEL,
    supabase_url=settings.SUPABASE_URL,
    supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY
)

logger.info("✅ Crafter agent initialized")

# Initialize UpdateDBPipeline
pipeline = UpdateDBPipeline(
    embedding_model=settings.EMBEDDING_MODEL,
    supabase_url=settings.SUPABASE_URL,
    supabase_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY,
    chunk_size=settings.CHUNK_SIZE
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
    try:
        logger.info(f"Received question: {request.query[:25]}...")

        # Call Crafter agent
        # print_to_cli=False to avoid console output in API
        result = crafter.ask(request.query, print_to_cli=False)

        logger.info(
            f"Successfully processed question, "
            f"returned {len(result['chunks'])} chunks"
        )

        return QuestionResponse(
            answer=result["answer"],
            retrieved_chunks=result["chunks"]
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
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

    except Exception as e:
        logger.error(f"Error starting database update: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start database update: {str(e)}"
        )


# Entry point for running with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)