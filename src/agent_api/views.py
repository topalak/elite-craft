"""
Django views for Elite Craft agent API.

This module replaces the FastAPI server and:
- Exposes REST endpoints for the Crafter agent
- Handles requests from Streamlit frontend
- Returns JSON responses
"""
import json
import logging

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from config import settings
from elite_craft.agent.crafter_agent import Crafter
from elite_craft.enums import Provider


logger = logging.getLogger(__name__)
logger.setLevel(settings.LOGGING_LEVEL)

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

logger.info("Crafter agent initialized")


@csrf_exempt
@require_POST
def ask(request):
    """
    Main endpoint: Ask the Crafter agent a question.

    This is the core functionality - users ask questions about
    building agents, and get answers based on retrieved documentation.

    Example:
        POST http://localhost:8000/api/ask
        Body: {"query": "How do I build an agent?"}

    Returns:
        JsonResponse with answer and retrieved chunks

    Status Codes:
        200: Success
        400: Empty or missing query
        504: LLM request timed out
        503: Cannot reach LLM service
        500: Internal server error
    """
    try:
        body = json.loads(request.body)
    except (json.JSONDecodeError, ValueError):
        return JsonResponse(
            {"detail": "Invalid JSON body"},
            status=400,
        )

    query = body.get("query", "").strip()
    if not query:
        logger.warning("Empty query received")
        return JsonResponse(
            {"detail": "Query cannot be empty"},
            status=400,
        )

    try:
        answer = crafter.ask(query=query, print_to_cli=False)
        return JsonResponse({
            "answer": answer,
            "retrieved_chunks": [],
        })

    except TimeoutError as e:
        logger.error(f"LLM request timeout: {e}")
        return JsonResponse(
            {"detail": "AI service request timed out. Please try again."},
            status=504,
        )

    except ConnectionError as e:
        logger.error(f"LLM connection failed: {e}")
        return JsonResponse(
            {"detail": "Cannot reach AI service. Please try again later."},
            status=503,
        )

    except Exception as e:
        logger.exception(f"Unexpected error in ask: {e}")
        if settings.DEBUG:
            detail = f"Error: {str(e)}"
        else:
            detail = "An internal error occurred. Please contact support if this persists."
        return JsonResponse({"detail": detail}, status=500)