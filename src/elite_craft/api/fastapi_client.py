"""
Client wrapper for Elite Craft FastAPI backend.

Example:
    ```python
    from elite_craft.api import EliteCraftClient

    # Local development
    client = EliteCraftClient(host="localhost", port=8000)

    # Using ngrok tunnel (e.g., from Colab)
    client = EliteCraftClient(
        use_ngrok=True,
        ngrok_url="https://xxxx-xx-xx-xx-xx.ngrok-free.app"
    )

    # Ask a question
    response = client.ask_question("How do I build an agent?")
    print(response.answer)

    # Update database
    result = client.update_database(["https://docs.langchain.com/..."])
    ```
"""
import requests

from elite_craft.api.schemas import (
    QuestionRequest,
    QuestionResponse,
    UpdateDBRequest,
    UpdateDBResponse,
)


class EliteCraftClient:
    """
    Client for Elite Craft API.

    Args:
        host: Elite Craft API host (e.g., "localhost" or "127.0.0.1")
        port: Elite Craft API port (e.g., 8000)
        use_ngrok: If True, uses ngrok_url instead of host:port
        ngrok_url: Full ngrok tunnel URL (e.g., "https://xxxx.ngrok-free.app")
    """
    def __init__(
        self,
        host: str = None,
        port: int | str = None,
        use_ngrok: bool = False,
        ngrok_url: str = "",
    ):
        """
        Initialize the API client.

        Args:
            host: Elite Craft API host (default: "localhost")
            port: Elite Craft API port (default: 8000)
            use_ngrok: If True, uses ngrok_url instead of host:port
            ngrok_url: Full ngrok tunnel URL including https://

        Raises:
            ValueError: If use_ngrok is True but ngrok_url is empty
        """
        if use_ngrok:
            if not ngrok_url:
                raise ValueError(
                    "ngrok_url must be provided when use_ngrok=True"
                )
            self.base_url = ngrok_url.rstrip("/")
        else:
            if not host or not port:
                raise ValueError(
                    "host and port must be provided whenuse_ngrok = False"
                )
            self.base_url = f"http://{host}:{port}"

    def ask_question(self, query: str) -> QuestionResponse:
        """
        Ask a question to the agent.

        Args:
            query: Question about agent development

        Returns:
            QuestionResponse with answer and retrieved chunks
        """
        request_data = QuestionRequest(query=query)
        response = requests.post(
            url=f"{self.base_url}/api/ask",
            json=request_data.model_dump(),
            timeout=90,
        )
        response.raise_for_status()
        return QuestionResponse(**response.json())

    def update_database(self, urls: list[str]) -> UpdateDBResponse:
        """
        Update database with new URLs.

        Args:
            urls: List of documentation URLs to add

        Returns:
            UpdateDBResponse with status
        """
        request_data = UpdateDBRequest(urls=urls)
        response = requests.post(
            url=f"{self.base_url}/api/update-db",
            json=request_data.model_dump(),
            timeout=10,
        )
        response.raise_for_status()
        return UpdateDBResponse(**response.json())