import datetime
import logging
import os

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# __file__ = current file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# os.pardir = os parent directory
# Python will look for .env file in parent folder
ENV_FILE_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Uses Pydantic Settings for automatic environment variable loading
    and validation. Loads from .env file in parent directory.

    Attributes:
        OLLAMA_API_KEY: API key for Ollama cloud service (SecretStr)
        GROQ_API_KEY: API key for Groq cloud service (SecretStr)
        SUPABASE_URL: Supabase project URL (SecretStr)
        SUPABASE_SERVICE_ROLE_SECRET_KEY: Supabase service role key (SecretStr)
        SUPABASE_ANON_PUBLIC_KEY: Supabase anonymous public key (SecretStr)
        LANGSMITH_TRACING: Enable/disable LangSmith tracing
        LANGSMITH_ENDPOINT: LangSmith API endpoint URL
        LANGSMITH_API_KEY: LangSmith API key for tracing (SecretStr)
        LANGSMITH_PROJECT: LangSmith project name
        EMBEDDING_MODEL: Name of embedding model (default: embeddinggemma)
        LLM_NAME: Name of LLM model (default: gpt-oss:20b-cloud)
        OUTPUT: Output directory path
        TIME_ZONE: Timezone for timestamps (default: UTC+3)
        LOGGING_LEVEL: Logging level (default: WARNING)
        DB_UPLOAD_BATCH_SIZE: Batch size for database uploads
        BODY_PREVIEW_END: Character limit for body preview column
        CHUNK_SIZE: Max chunk size per chunk
        API_HOST: API server host (default: localhost)
        API_PORT: API server port (default: 8000)
    """
    OLLAMA_API_KEY: SecretStr = ""
    OLLAMA_HOST_LOCAL: SecretStr = ""
    OLLAMA_HOST_COLAB: SecretStr = ""
    OLLAMA_COLAB_API_KEY: SecretStr = ""
    GROQ_API_KEY: SecretStr = ""
    SUPABASE_URL: SecretStr = ""
    SUPABASE_SERVICE_ROLE_SECRET_KEY: SecretStr = ""
    SUPABASE_ANON_PUBLIC_KEY: SecretStr = ""
    TAVILY_API_KEY: SecretStr = ""
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: SecretStr = ""
    LANGSMITH_PROJECT: str = "elite-craft"

    OUTPUT: str = os.path.join(ENV_FILE_DIR, 'out')
    TIME_ZONE: datetime.timezone = datetime.timezone(
        offset=datetime.timedelta(hours=3),
        name='UTC+3'
    )

    EMBEDDING_MODEL: str = "nomic-embed-text:v1.5"
    LLM_NAME: str = "gpt-oss:20b-cloud"
    #LLM_NAME: str = "ministral-3:14b-cloud"
    #LLM_NAME: str = "devstral-small-2:24b-cloud"
    #LLM_API_KEY: str = "nemotron-3-nano:30b-cloud"


    # API server configuration
    # These variables must be set
    API_HOST: str = "localhost"
    API_PORT: int = 8000

    # API timeout configuration (in seconds)
    API_REQUEST_TIMEOUT: int = Field(
        default=90,
        description="Timeout for API requests to agent endpoints (ask_question)"
    )
    API_UPDATE_DB_TIMEOUT: int = Field(
        default=100,
        description="Timeout for database update endpoint requests"
    )

    # Security configuration
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode - exposes detailed error messages (NEVER use in production!)"
    )

    # CORS configuration
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:8501", "http://localhost:8502"],
        description="Allowed CORS origins. Use ['*'] only in development!"
    )

    USE_OLLAMA_LOCAL: bool = False

    # Thread number for agent's memory
    DEFAULT_THREAD_ID: str = "1"

    LOGGING_LEVEL: int = logging.INFO
    #LOGGING_LEVEL: int = logging.WARNING

    # Database upload configuration
    DB_UPLOAD_BATCH_SIZE: int = 100

    # Embedding batch size (chunks per API call)
    # Lower values = more API calls but safer for large documents
    # Higher values = fewer API calls but may exceed context limits
    EMBEDDING_BATCH_SIZE: int = 20

    # Database "body_preview" column's preview size
    BODY_PREVIEW_END: int = 3000

    # WARNING: Changing CHUNK_SIZE requires re-ingesting ALL documents
    # This value affects knowledge base quality. Test retrieval before production.
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Default chunk size in characters. "

    )

    MINIMUM_CHUNK_SIZE: int = Field(
        default=500,
        description="Minimum chunk size in characters. "
    )

    CHUNK_OVERLAP: int = Field(
    default=300,
        description=(
            "Overlap value for each chunk. "
        )
    )

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file_encoding="utf-8",
        env_file=os.path.join(ENV_FILE_DIR, '.env')
    )


settings = Settings()
