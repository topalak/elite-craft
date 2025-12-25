# Elite Craft

**Forging Elite AI Systems**

An AI-powered assistant that helps developers build and enhance agentic AI projects using LangChain, LangGraph, and related frameworks.

---

## Project Structure

```
elite-craft/
├── src/
│   ├── config.py                              # Pydantic settings management
│   ├── main_dev.py                            # Development entry point for Crafter agent
│   └── elite_craft/
│       ├── __init__.py
│       ├── model_provider.py                  # LLM & embedding model configuration
│       ├── agent/                             # Agent implementation
│       │   ├── __init__.py
│       │   └── crafter_agent.py               # Main RAG agent with LLM and retriever
│       ├── tools/                             # Agent tools
│       │   ├── __init__.py
│       │   ├── retriever.py                   # Semantic search retriever with pgvector
│       │   └── handler.py                     # Tool factory with dependency injection
│       ├── api/                               # REST API layer
│       │   ├── __init__.py
│       │   ├── fastapi_server.py              # FastAPI server with endpoints
│       │   └── schemas.py                     # Pydantic request/response models
│       ├── frontend/                          # User interface
│       │   └── app.py                         # Streamlit chat interface
│       ├── services/                          # Core pipeline services
│       │   ├── __init__.py
│       │   ├── crawling.py                    # Async web crawling (Crawl4AI)
│       │   ├── chunking.py                    # Document chunking (Docling)
│       │   ├── embedding.py                   # Text embeddings (Ollama/HuggingFace)
│       │   ├── database_uploading.py          # Supabase upload operations
│       │   └── update_db_pipeline.py          # End-to-end ingestion pipeline
│       └── database/                          # Database schema
│           ├── db_table_setup.sql             # PostgreSQL tables with pgvector
│           └── db_cosine_similarity_function.sql  # Semantic search function
├── tests/                                     # Unit tests (100% coverage)
│   ├── test_chunking.py
│   ├── test_crawling.py
│   ├── test_database_uploading.py
│   ├── test_embedding.py
│   ├── test_model_provider.py
│   ├── test_update_db_pipeline.py
│   ├── are_crawling_outputs_stochastic.py        # Crawl stability analysis
│   └── are_db_chunks_and_crawled_chunks_same.py  # Database validation
├── .env.example                                  # Environment variables template
├── pyproject.toml
└── README.md
```

## Overview

Elite Craft is a RAG-powered assistant specialized in AI agent development. It provides:
- **Knowledge Base Management**: Crawl and process documentation from LangChain, LangGraph, Pydantic, and Supabase
- **Semantic Search**: Vector-based retrieval using pgvector and embeddings
- **Intelligent Responses**: Context-aware answers to technical questions about agent frameworks
- **Development Assistance**: Best practices guidance for building agentic AI systems

---

## Architecture

### Technology Stack

**Orchestration & Agent Framework:**
- **LangChain**: Core framework for LLM application development

**Data Processing:**
- **Crawl4AI**: Asynchronous web crawling with markdown conversion
- **Docling**: Intelligent document chunking with hybrid strategies
- **Pydantic**: Data validation and settings management

**Vector Database & Search:**
- **Supabase**: PostgreSQL with pgvector extension for vector storage
- **pgvector**: Cosine similarity search for semantic retrieval

**Embeddings & LLMs:**
- **Ollama**: Local embedding models (nomic-embed-text:v1.5)
- **HuggingFace**: Alternative embedding providers
- Configurable LLM providers (Ollama, Groq)

### Core Components

**Knowledge Base Pipeline:**
1. **UpdateDBPipeline** (`services/update_db_pipeline.py`): Orchestrates the complete document ingestion workflow
2. **Crawler** (`services/crawling.py`): Fetches web content and converts to markdown
3. **Chunker** (`services/chunking.py`): Splits documents into semantic chunks
4. **Embedder** (`services/embedding.py`): Generates vector embeddings for chunks
5. **DatabaseUploader** (`services/database_uploading.py`): Manages Supabase operations

**Agent System:**
6. **Retriever** (`tools/retriever.py`): Performs semantic search with source filtering
7. **Handler** (`tools/handler.py`): Factory for creating LangChain tools with proper dependency injection
8. **Crafter Agent** (`agent/crafter_agent.py`): RAG-powered agent with strict anti-hallucination guardrails

**Application Layer:**
9. **FastAPI Server** (`api/fastapi_server.py`): REST API exposing `/api/ask` and `/api/update-db` endpoints
10. **Streamlit App** (`frontend/app.py`): User-friendly chat interface for querying the agent

---

## Setup

### Prerequisites

- Python 3.13+
- PostgreSQL with pgvector extension
- Ollama (for local embeddings)
- Supabase account

### Installation

1. Clone the repository:
```bash
git clone https://github.com/topalak/elite-craft.git
cd elite-craft
```

2. Install dependencies:
```bash
uv sync
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. Set up the database:
```sql
-- Run the schema in Supabase SQL Editor
-- File: src/elite_craft/database/db_setup.sql
```

---

## Configuration

Create a `.env` file in the project root (use `.env.example` as template):

```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_SECRET_KEY=your-service-role-key

# LangSmith Configuration (Optional - for tracing)
LANGSMITH_API_KEY=your-api-key
LANGSMITH_TRACING=false

# LLM Provider API Keys
OLLAMA_API_KEY=your-ollama-key
```

The configuration is managed through Pydantic Settings in `src/config.py` with the following defaults:
- Embedding model: `nomic-embed-text:v1.5`
- LLM: `gpt-oss:20b-cloud`
- Batch size for database uploads: 100
- Timezone: UTC+3

---

## Running the Application

Elite Craft has two main components that run as separate services:

### 1. FastAPI Backend Server

The backend exposes REST endpoints for the Crafter agent:

```bash
# Start the FastAPI server (from project root)
PYTHONPATH=./src uvicorn elite_craft.api.fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

**Access:**
- API Server: `http://localhost:8000`
- Interactive API Documentation: `http://localhost:8000/docs`
- Alternative Documentation: `http://localhost:8000/redoc`

**Note:** On first startup, Ollama will download the model (`qwen2.5-coder:7b`, ~4.68GB). The server won't be fully functional until this completes.

### 2. Streamlit Frontend

The frontend provides a user-friendly chat interface:

```bash
# Start the Streamlit app (from project root)
PYTHONPATH=./src streamlit run src/elite_craft/frontend/app.py
```

**Access:**
- Local URL: `http://localhost:8502`

**Important:** Both services require `PYTHONPATH` to be set because `config.py` is located in the `src/` directory.

### Running Both Services

Open two terminal windows and run each command in a separate terminal:

```bash
# Terminal 1 - Backend
PYTHONPATH=./src uvicorn elite_craft.api.fastapi_server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend
PYTHONPATH=./src streamlit run src/elite_craft/frontend/app.py
```

To stop the services, press `CTRL+C` in each terminal.

### Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'config'` or `No module named 'elite_craft'`**

This happens when `PYTHONPATH` is not set correctly. Make sure you're running the commands from the project root directory and include `PYTHONPATH=./src` before each command.

**Issue: Ollama model downloading on first startup**

The FastAPI server will download the `qwen2.5-coder:7b` model (~4.68GB) on first startup. This is normal and only happens once. Wait for the download to complete before making API requests.

**Long-term fix for PYTHONPATH:**

To avoid needing `PYTHONPATH` every time, you can:
1. Install the package in editable mode: `pip install -e .` (requires `setup.py` or proper `pyproject.toml` configuration)
2. Move `config.py` into the `elite_craft/` package
3. Add `export PYTHONPATH=./src` to your shell profile (`.bashrc`, `.zshrc`, etc.)

---

## Usage

### 1. Building the Knowledge Base

Process documentation URLs and populate your vector database:

```python
import asyncio
from elite_craft.services import UpdateDBPipeline


async def main():
    pipeline = UpdateDBPipeline()

    urls = [
        "https://docs.langchain.com/oss/python/langgraph/overview",
        "https://docs.langchain.com/oss/python/langgraph/tutorials/introduction",
    ]

    results = await pipeline.process_multiple_urls(urls)
    print(f"Processed {len(results)} URLs")


if __name__ == "__main__":
    asyncio.run(main())
```

## Pipeline Flow

```
URL → Crawl → Upload Metadata → Chunk → Embed → Upload Chunks → Complete
```

1. **Crawl**: Fetch and convert HTML to Markdown
2. **Upload Metadata**: Store URL, source, timestamp, preview
3. **Chunk**: Split content into manageable pieces
4. **Embed**: Generate vector embeddings
5. **Upload Chunks**: Store chunks with embeddings for retrieval

---

## Database Schema

### Metadata Table
- `id`: Serial primary key
- `url`: Unique URL identifier
- `source`: Framework name (langchain, docling, etc.)
- `crawled_time`: Timestamp with timezone
- `body_preview`: Text preview (300-3000 chars)

### Chunks Table
- `id`: Serial primary key
- `url`: Foreign key to metadata
- `chunk_number`: Chunk sequence number
- `content`: Chunk text
- `embedding`: Vector(768) for similarity search
- `created_at`: Timestamp

---

## Testing

Elite Craft maintains **100% test coverage** for all core services with comprehensive unit tests.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov

# Run specific test file
pytest tests/test_crawling.py -v

# Run with detailed coverage report
pytest --cov --cov-report=html
```

### Test Coverage

**Core Services - 100% Coverage:**
- ✅ `config.py` - Configuration and settings management
- ✅ `model_provider.py` - LLM and embedding model providers
- ✅ `services/crawling.py` - Web crawling and source extraction
- ✅ `services/chunking.py` - Document chunking logic
- ✅ `services/embedding.py` - Embedding generation
- ✅ `services/database_uploading.py` - Database operations
- ✅ `services/update_db_pipeline.py` - End-to-end pipeline orchestration

**Overall Project Coverage: ~90%**

### Test Structure

```
tests/
├── test_chunking.py                         # Document chunking tests
├── test_crawling.py                         # Web crawling and source mapping
├── test_database_uploading.py               # Supabase operations
├── test_embedding.py                        # Embedding generation
├── test_model_provider.py                   # Model configuration (Ollama, Groq)
├── test_update_db_pipeline.py               # Pipeline integration tests
└── are_crawling_outputs_stochastic.py       # Crawl stability analysis script
└── are_db_chunks_and_crawled_chunks_same.py # Checks database and crawled chunks
```

### Crawl Stability Analysis

The `are_crawling_outputs_stochastic.py` script is an analysis tool (not a unit test) that checks whether web crawling produces deterministic outputs:
The `are_db_chunks_and_crawled_chunks_same.py` scripy is an analysis tool  (not a unit test) that checks database chunks and crawled chunks are same?
**Purpose:**
- Detects if crawled content changes between requests
- Identifies AI-generated responses in documentation sites
- Analyzes text differences with multiple normalization strategies
- Helps ensure data quality for the knowledge base

**Usage:**
```bash
# Run stability analysis on URLs
python tests/are_crawling_outputs_stochastic.py
python tests/are_db_chunks_and_crawled_chunks_same.py
```

**Features:**
- Multiple crawl attempts per URL with retry logic
- Text similarity analysis (raw, whitespace-normalized, line-normalized)
- Character-level diff reporting using `difflib.SequenceMatcher`
- Detection of dynamic content (e.g., "AI-generated responses" warnings)
- Detailed statistics and difference reports

**Analysis Metrics:**
- Raw similarity percentage
- Similarity after space normalization
- Similarity without whitespace
- Line count differences
- Character-level changes (added, removed, modified)

### Testing Best Practices

All tests follow professional unit testing standards:

1. **Isolation**: External dependencies (APIs, databases, models) are mocked
2. **Async Support**: Full support for async/await patterns with `pytest-asyncio`
3. **Mock Patterns**: Proper use of `Mock`, `AsyncMock`, and `patch` for dependencies
4. **Coverage**: Every code path is tested including error handling
5. **Clear Documentation**: Each test has descriptive docstrings

**Example Test Pattern:**
```python
async def test_crawl_known_source(self):
    """Test that known domains return correct source name."""

    with patch('module.AsyncWebCrawler') as MockCrawler:
        # Setup mock behavior
        mock_instance = MockCrawler.return_value.__aenter__.return_value
        mock_instance.arun.return_value = Mock(markdown="# Content")

        # Execute
        result = await crawl("https://docs.langchain.com/guide")

        # Assert
        assert result["source"] == "langchain"
```

---

## Contributing

Contributions are welcome! Please ensure:

1. Code follows `CLAUDE.md` standards
2. All functions have proper docstrings
3. Error handling is comprehensive
4. Logging is used (not print statements)
5. Type hints are included
6. **Tests are written for new code** - Maintain 100% coverage for core services
7. All tests pass: `pytest --cov`

---

## Acknowledgments

- Built with [LangChain](https://docs.langchain.com/)
- Powered by [Supabase](https://supabase.com/)
- Document processing by [Docling](https://docling-project.github.io/)
- Web crawling by [Crawl4AI](https://docs.crawl4ai.com/)