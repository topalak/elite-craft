import datetime
import logging
import os

from pydantic import Field
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
        OLLAMA_API_KEY: API key for Ollama cloud service
        GROQ_API_KEY: API key for Groq cloud service
        SUPABASE_URL: Supabase project URL
        SUPABASE_SERVICE_ROLE_SECRET_KEY: Supabase service role key
        SUPABASE_ANON_PUBLIC_KEY: Supabase anonymous public key
        LANGSMITH_TRACING: Enable/disable LangSmith tracing
        LANGSMITH_ENDPOINT: LangSmith API endpoint URL
        LANGSMITH_API_KEY: LangSmith API key for tracing
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
    OLLAMA_API_KEY: str = ""
    OLLAMA_HOST_LOCAL: str = ""
    OLLAMA_HOST_COLAB: str = ""
    OLLAMA_COLAB_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_SECRET_KEY: str = ""
    SUPABASE_ANON_PUBLIC_KEY: str = ""
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "elite-craft"

    OUTPUT: str = os.path.join(ENV_FILE_DIR, 'out')
    TIME_ZONE: datetime.timezone = datetime.timezone(
        offset=datetime.timedelta(hours=3),
        name='UTC+3'
    )

    EMBEDDING_MODEL: str = "nomic-embed-text:v1.5"
    #LLM_NAME: str = "ministral-3:14b-cloud"
    #LLM_NAME: str = "qwen3-coder:30b"
    LLM_NAME: str = "gpt-oss:20b-cloud"

    # API server configuration
    # Must use these variable
    API_HOST: str = "localhost"
    API_PORT: int = 8000

    USE_OLLAMA_LOCAL: bool = False

    #LOGGING_LEVEL: int = logging.WARNING
    LOGGING_LEVEL: int = logging.INFO

    # Database upload configuration
    DB_UPLOAD_BATCH_SIZE: int = 100

    # Database "body_preview" column's preview size
    BODY_PREVIEW_END: int = 3000

    # WARNING: Changing CHUNK_SIZE requires re-ingesting ALL documents
    # This value affects knowledge base quality. Test retrieval before production.
    CHUNK_SIZE: int = Field(
        default=1000,
        ge=100,
        le=8000,
        description=(
            "Maximum chunk size in characters for document ingestion. "
        )
    )
    CHUNK_OVERLAP: int = Field(
    default=200,
        ge=20,
        le=1500,
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
