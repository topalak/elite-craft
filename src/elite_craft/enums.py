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