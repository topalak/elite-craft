import datetime
import logging
import os

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
        LANGSMITH_API_KEY: LangSmith API key for tracing
        LANGSMITH_TRACING: Enable/disable LangSmith tracing
        EMBEDDING_MODEL: Name of embedding model (default: embeddinggemma)
        LLM_NAME: Name of LLM model (default: gpt-oss:20b-cloud)
        OUTPUT: Output directory path
        TIME_ZONE: Timezone for timestamps (default: UTC+3)
        LOGGING_LEVEL: Logging level (default: WARNING)
        DB_UPLOAD_BATCH_SIZE: Batch size for database uploads
        BODY_PREVIEW_END: Character limit for body preview column
        API_HOST: API server host (default: localhost)
        API_PORT: API server port (default: 8000)
    """
    OLLAMA_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_SERVICE_ROLE_SECRET_KEY: str = ""
    SUPABASE_ANON_PUBLIC_KEY: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_TRACING: str = "false"

    EMBEDDING_MODEL: str = "nomic-embed-text:v1.5"
    #EMBEDDING_MODEL: str = "embeddinggemma"
    MAX_TOKEN_SIZE: int = 512
    LLM_NAME: str = "gpt-oss:20b-cloud"

    OUTPUT: str = os.path.join(ENV_FILE_DIR, 'out')
    TIME_ZONE: datetime.timezone = datetime.timezone(
        offset=datetime.timedelta(hours=3),
        name='UTC+3'
    )

    #LOGGING_LEVEL: int = logging.WARNING
    LOGGING_LEVEL: int = logging.INFO

    # Database upload configuration
    DB_UPLOAD_BATCH_SIZE: int = 100

    # Database "body_preview" column's preview size
    BODY_PREVIEW_END: int = 3000

    # API server configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8000

    model_config = SettingsConfigDict(
        extra="ignore",
        env_file_encoding="utf-8",
        env_file=os.path.join(ENV_FILE_DIR, '.env')
    )


settings = Settings()
