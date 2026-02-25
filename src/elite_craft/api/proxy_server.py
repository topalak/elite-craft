"""
Simple proxy server for LLM API access from sandboxed code.

The real API keys stay in the proxy, sandboxed code only knows the proxy URL.
"""

from fastapi import FastAPI, Request, HTTPException, Header
from config import settings

app = FastAPI()

# Load config
PROXY_SECRET = settings.PROXY_SECRET.get_secret_value()
LLM_PROVIDER = settings.LLM_PROVIDER
LLM_MODEL = settings.LLM_NAME


def _require_key(field_name: str) -> str:
    """Get a secret value from settings, raising if empty."""
    value = getattr(settings, field_name).get_secret_value()
    if not value:
        raise ValueError(
            f"Provider '{LLM_PROVIDER}' requires {field_name} to be set in .env"
        )
    return value


def _get_llm_client():
    """Get LLM client based on settings."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=LLM_MODEL,
            api_key=_require_key("GROQ_API_KEY"),
            temperature=0
        )

    elif LLM_PROVIDER == "ollama_cloud":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LLM_MODEL,
            base_url="https://ollama.com",
            client_kwargs={
                'headers': {'Authorization': f'Bearer {_require_key("OLLAMA_API_KEY")}'}
            },
            temperature=0
        )

    elif LLM_PROVIDER == "ollama_local":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LLM_MODEL,
            base_url=settings.OLLAMA_HOST_LOCAL.get_secret_value() or "http://localhost:11434",
            temperature=0
        )

    else:
        raise ValueError(f"Unsupported provider: {LLM_PROVIDER}")

# Initialize LLM client
llm_client = _get_llm_client()

@app.post("/api/chat")
async def ollama_chat(request: Request, authorization: str = Header(None)):
    """
    Ollama-compatible /api/chat endpoint.

    Makes the proxy act like an Ollama server for ChatOllama clients.
    Forwards requests to the real LLM backend (which has the API key).
    """
    # Check authentication
    if authorization != f"Bearer {PROXY_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Get Ollama request format
    body = await request.json()
    messages = body.get("messages", [])

    # Call backend LLM (has real API key configured)
    response = llm_client.invoke(messages)

    # Return Ollama response format
    return {
        "model": body.get("model", LLM_MODEL),
        "created_at": "2024-01-01T00:00:00Z",
        "message": {
            "role": "assistant",
            "content": response.content
        },
        "done": True
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "provider": LLM_PROVIDER}


if __name__ == "__main__":
    import uvicorn
    print(f"Starting proxy on http://0.0.0.0:4000 (provider: {LLM_PROVIDER})")
    uvicorn.run(app, host="0.0.0.0", port=4000)