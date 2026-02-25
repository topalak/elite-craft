from enum import StrEnum

class GeneralEnums(StrEnum):
    """
    General purpose enumerations for common field names and table identifiers.

    This enum provides a centralized reference for commonly used string literals
    across the application, reducing typos and improving code maintainability.

    Attributes:
        BODY_TEXT: Field name for document body text content
        BODY_PREVIEW: Field name for document body preview content
        DOCUMENTS: Database table name for documents
        CHUNKS: Database table name for text chunks
    """

    BODY_TEXT = "body_text"
    BODY_PREVIEW = "body_preview"
    DOCUMENTS = "documents"
    CHUNKS = "chunks"


class Provider(StrEnum):
    """
    Supported LLM providers.

    Attributes:
        OLLAMA_LOCAL: Local Ollama instance (requires model_provider_url)
        OLLAMA_CLOUD: Ollama Cloud API (requires api_key)
        GROQ: Groq Cloud API (requires api_key)
    """
    OLLAMA_LOCAL = "ollama_local"
    OLLAMA_CLOUD = "ollama_cloud"
    GROQ = "groq"
    OPENAI = "openai"

class Model(StrEnum):

    GPT_OSS_20 = "gpt-oss:20b-cloud"
    GPT_OSS_120 = "gpt-oss:120b-cloud"
    KIMI_K2_5 = "kimi-k2-5"

    NOMIC = "nomic-embed-text:v1.5"