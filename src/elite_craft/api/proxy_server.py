"""
Simple proxy server for LLM API access from sandboxed code.

The real API keys stay in the proxy, sandboxed code only knows the proxy URL.
"""

from fastapi import FastAPI, Request, HTTPException, Header
from config import settings

app = FastAPI()

# Load config
PROXY_SECRET = settings.PROXY_SECRET
LLM_PROVIDER = settings.LLM_PROVIDER
LLM_MODEL = settings.LLM_NAME


def _get_llm_client():
    """Get LLM client based on settings."""
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=LLM_MODEL,
            api_key=settings.GROQ_API_KEY.get_secret_value(),
            temperature=0
        )

    elif LLM_PROVIDER == "ollama_cloud":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LLM_MODEL,
            base_url="https://ollama.com",
            client_kwargs={
                'headers': {'Authorization': f'Bearer {settings.OLLAMA_API_KEY.get_secret_value()}'}
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


@app.post("/v1/chat/completions")
async def proxy_chat(request: Request, authorization: str = Header(None)):
    """
    Proxy endpoint for LLM chat completions.

    Requires: Authorization header with PROXY_SECRET
    """
    # Check authentication
    if authorization != f"Bearer {PROXY_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Get request body
    body = await request.json()
    messages = body.get("messages", [])

    # Call LLM
    response = llm_client.invoke(messages)

    # Return OpenAI-compatible response
    return {
        "id": "chatcmpl-proxy",
        "object": "chat.completion",
        "model": body.get("model", LLM_MODEL),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response.content
            },
            "finish_reason": "stop"
        }]
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "provider": LLM_PROVIDER}


if __name__ == "__main__":
    import uvicorn
    print(f"Starting proxy on http://0.0.0.0:4000 (provider: {LLM_PROVIDER})")
    uvicorn.run(app, host="0.0.0.0", port=4000)