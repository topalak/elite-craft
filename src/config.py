import datetime
import logging
import os

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from elite_craft.enums import Model, Provider

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
    """
    #OPTIONAL
    OLLAMA_API_KEY: SecretStr = ""
    OLLAMA_HOST_LOCAL: SecretStr = ""
    OLLAMA_HOST_COLAB: SecretStr = ""
    OLLAMA_COLAB_API_KEY: SecretStr = ""
    GROQ_API_KEY: SecretStr = ""
    OPENAI_API_KEY: SecretStr = ""

    #REQUIRED, MUST BE SET IN .ENV
    SUPABASE_URL: SecretStr
    SUPABASE_SERVICE_ROLE_SECRET_KEY: SecretStr
    SUPABASE_ANON_PUBLIC_KEY: SecretStr
    TAVILY_API_KEY: SecretStr
    PROXY_SECRET: SecretStr

    OUTPUT: str = os.path.join(ENV_FILE_DIR, 'out')
    TIME_ZONE: datetime.timezone = datetime.timezone(
        offset=datetime.timedelta(hours=3),
        name='UTC+3'
    )

    # AGENT'S LLM CONFIGURATION
    LLM_NAME: str = Model.KIMI_K2_5
    LLM_PROVIDER: str = Provider.OLLAMA_CLOUD




    # Security configuration
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode - exposes detailed error messages"
    )


    USE_OLLAMA_LOCAL: bool = False

    # Thread number for agent's memory
    DEFAULT_THREAD_ID: str = "1"

    LOGGING_LEVEL: int = logging.INFO
    #LOGGING_LEVEL: int = logging.WARNING





    # SANDBOX CONFIGURATION
    SANDBOX_LLM_NAME: str = Model.GPT_OSS_20

    # SERVICES CONFIGURATION
    EMBEDDING_MODEL: str = Model.NOMIC
    DB_UPLOAD_BATCH_SIZE: int = 100
    EMBEDDING_BATCH_SIZE: int = 20
    BODY_PREVIEW_END: int = 3000
    CHUNK_SIZE: int = Field(default=1000,description="Default chunk size in characters.")
    MINIMUM_CHUNK_SIZE: int = Field(default=500,description="Minimum chunk size in characters.")
    CHUNK_OVERLAP: int = Field(default=200,description="Overlap value for each chunk.")

    # LANGSMITH CONFIGURATION
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: SecretStr = ""
    LANGSMITH_PROJECT: str = "elite-craft"

    # API SERVER CONFIGURATION
    # @2 variables must be set
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_REQUEST_TIMEOUT: int = Field(
        default=300,
        description="Timeout for API requests to agent endpoints (ask_question)")
    API_UPDATE_DB_TIMEOUT: int = Field(
        default=100,
        description="Timeout for database update endpoint requests")
    #todo update the cors for django
    # CORS configuration
    ALLOWED_ORIGINS: list[str] = Field(
        default=["http://localhost:8501", "http://localhost:8502"],
        description="Allowed CORS origins. Use ['*'] only in development!"
    )


    model_config = SettingsConfigDict(extra="ignore",env_file_encoding="utf-8",env_file=os.path.join(ENV_FILE_DIR, '.env'))


settings = Settings()
