# Elite Craft

**Forging Elite AI Systems**

An AI-powered coding assistant that helps developers build and enhance agentic AI projects using LangChain, LangGraph, and DeepAgents frameworks.

---

## Project Structure

```
elite-craft/
├── src/
│   ├── config.py                              # Pydantic settings management
│   ├── main_dev.py                            # Development entry point for Crafter agent
│   └── elite_craft/
│       ├── __init__.py
│       ├── enums.py                           # Provider and general enumerations
│       ├── model_provider.py                  # LLM & embedding model configuration
│       ├── agent/                             # Agent implementation
│       │   ├── __init__.py
│       │   └── crafter_agent.py               # Code generation agent with web search & sandbox
│       ├── tools/                             # Agent tools
│       │   ├── __init__.py
│       │   ├── retriever.py                   # Semantic search retriever with pgvector
│       │   ├── web_search.py                  # Real-time web search via Tavily API
│       │   ├── handler.py                     # Tool factory with dependency injection
│       │   └── code_executor/                 # Sandboxed code execution
│       │       ├── __init__.py
│       │       └── executor.py                # Docker-based Python sandbox
│       ├── api/                               # REST API layer
│       │   ├── __init__.py
│       │   ├── fastapi_server.py              # FastAPI server with endpoints
│       │   ├── fastapi_client.py              # Python client for API access
│       │   ├── proxy_server.py                # Proxy for sandbox LLM access
│       │   └── schemas.py                     # Pydantic request/response models
│       ├── frontend/                          # User interface
│       │   ├── __init__.py
│       │   └── app.py                         # Streamlit chat interface
│       ├── services/                          # Core pipeline services
│       │   ├── __init__.py
│       │   ├── schemas.py                     # Service-layer Pydantic models
│       │   ├── crawling.py                    # Async web crawling (Crawl4AI)
│       │   ├── chunking.py                    # Document chunking (LangChain splitters)
│       │   ├── embedding.py                   # Text embeddings (Ollama)
│       │   ├── database_uploading.py          # Supabase upload operations
│       │   └── update_db_pipeline.py          # End-to-end ingestion pipeline
│       └── database/                          # Database schema
│           ├── db_table_setup.sql             # PostgreSQL tables with pgvector
│           └── db_cosine_similarity_function.sql  # Semantic search function
├── tests/                                     # Unit tests
│   ├── services/                              # Service layer tests
│   │   ├── test_chunking.py
│   │   ├── test_crawling.py
│   │   ├── test_database_uploading.py
│   │   ├── test_embedding.py
│   │   ├── test_model_provider.py
│   │   └── test_update_db_pipeline.py
│   ├── test_retrieval_quality.py              # Retrieval quality tests
│   ├── generated_code_verification.py         # Code generation verification
│   ├── are_crawling_outputs_stochastic.py     # Crawl stability analysis
│   └── are_db_chunks_and_crawled_chunks_same.py  # Database validation
├── .env.example                               # Environment variables template
├── pyproject.toml
└── README.md
```

## Overview

Elite Craft is a code generation assistant specialized in AI agent development. It provides:
- **Code Generation**: Generate production-ready Python code for agentic AI systems
- **Sandboxed Execution**: Test all generated code in isolated Docker containers
- **Real-time Web Search**: Access current documentation via Tavily API
- **Knowledge Base**: Semantic search over LangChain, LangGraph, and DeepAgents documentation
- **Anti-Hallucination**: Strict guardrails ensure framework-specific code is always verified

---

## Architecture

### Technology Stack

**Orchestration & Agent Framework:**
- **LangChain**: Core framework for LLM application development
- **DeepAgents**: Advanced agent patterns built on LangGraph
- **TodoListMiddleware**: Task planning for complex multi-step code generation

**Agent Tools:**
- **Tavily**: Real-time web search for current documentation and API references
- **Docker**: Sandboxed Python code execution with security controls
- **pgvector**: Semantic search over knowledge base (optional retriever tool)

**Data Processing:**
- **Crawl4AI**: Asynchronous web crawling with markdown conversion
- **LangChain Text Splitters**: RecursiveCharacterTextSplitter with code block handling
- **Pydantic**: Data validation and settings management with SecretStr for sensitive data

**Vector Database & Search:**
- **Supabase**: PostgreSQL with pgvector extension for vector storage
- **pgvector**: Cosine similarity search for semantic retrieval

**Embeddings & LLMs:**
- **Ollama**: Embedding models (nomic-embed-text:v1.5 - 768 dimensions)
- Configurable LLM providers via `Provider` enum:
  - `Provider.OLLAMA_CLOUD`: Ollama Cloud API (default)
  - `Provider.OLLAMA_LOCAL`: Local Ollama instance
  - `Provider.GROQ`: Groq Cloud API

### Core Components

**Agent System:**
1. **Crafter Agent** (`agent/crafter_agent.py`): Code generation agent with web search, sandbox execution, and todo planning
2. **Handler** (`tools/handler.py`): Factory for creating LangChain tools with dependency injection
3. **WebSearch** (`tools/web_search.py`): Tavily-powered real-time web search
4. **CodeExecutor** (`tools/code_executor/executor.py`): Docker-based Python sandbox with security controls
5. **Retriever** (`tools/retriever.py`): Semantic search against documentation knowledge base

**Knowledge Base Pipeline:**
6. **UpdateDBPipeline** (`services/update_db_pipeline.py`): Orchestrates document ingestion workflow
7. **Crawler** (`services/crawling.py`): Fetches web content and converts to markdown
8. **Chunker** (`services/chunking.py`): Document chunking with minimum size enforcement
9. **Embedder** (`services/embedding.py`): Batched embedding generation
10. **DatabaseUploader** (`services/database_uploading.py`): Supabase operations with batch inserts

**Application Layer:**
11. **FastAPI Server** (`api/fastapi_server.py`): REST API exposing `/api/ask` and `/api/update-db` endpoints
12. **FastAPI Client** (`api/fastapi_client.py`): Python client for programmatic API access
13. **Proxy Server** (`api/proxy_server.py`): Secure proxy for sandbox LLM access
14. **Streamlit App** (`frontend/app.py`): User-friendly chat interface

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

3. Install the package in editable mode:
```bash
uv pip install -e .
```

This allows you to run all commands without setting `PYTHONPATH` every time. Changes to the code are immediately available without reinstalling.

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

5. Set up the database:
```sql
-- Run these SQL files in Supabase SQL Editor in order:
-- 1. src/elite_craft/database/db_table_setup.sql (creates tables)
-- 2. src/elite_craft/database/db_cosine_similarity_function.sql (creates search function)
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
GROQ_API_KEY=your-groq-key  # Optional: for Groq provider

# Web Search
TAVILY_API_KEY=your-tavily-key

# Code Executor Proxy
PROXY_SECRET=your-proxy-secret  # For sandbox LLM access
```

The configuration is managed through Pydantic Settings in `src/config.py` with the following defaults:
- Embedding model: `nomic-embed-text:v1.5` (768-dimensional vectors)
- LLM: `gpt-oss:120b-cloud` (Ollama Cloud)
- LLM Provider: `ollama_cloud` (options: `ollama_cloud`, `ollama_local`, `groq`)
- Chunk size: 1000 characters with 200 character overlap
- Minimum chunk size: 500 characters
- Batch size for database uploads: 100
- Embedding batch size: 20 chunks per API call
- API request timeout: 90 seconds
- Timezone: UTC+3
- Security: All API keys and secrets use Pydantic SecretStr for enhanced protection

---

## Running the Application

Elite Craft has two main components that run as separate services:

### 1. FastAPI Backend Server

The backend exposes REST endpoints for the Crafter agent:

```bash
# Start the FastAPI server (from project root)
uvicorn elite_craft.api.fastapi_server:app --host 0.0.0.0 --port 8000 --reload
```

**Access:**
- API Server: `http://localhost:8000`
- Interactive API Documentation: `http://localhost:8000/docs`
- Alternative Documentation: `http://localhost:8000/redoc`

**Note:** On first startup, Ollama may need to download the embedding model (`nomic-embed-text:v1.5`, ~274MB) and LLM model (`ministral-3:8b-cloud`). The server won't be fully functional until these downloads complete.

### 2. Streamlit Frontend

The frontend provides a user-friendly chat interface:

```bash
# Start the Streamlit app (from project root)
streamlit run src/elite_craft/frontend/app.py
```

**Access:**
- Local URL: `http://localhost:8502`

### 3. Proxy Server (Optional)

The proxy server provides secure API access for sandboxed code execution:

```bash
# Start the proxy server (from project root)
uvicorn elite_craft.api.proxy_server:app --host 0.0.0.0 --port 4000 --reload
```

**Access:**
- Proxy Server: `http://localhost:4000`
- Health Check: `http://localhost:4000/health`

### Running Multiple Services

Open separate terminal windows for each service:

```bash
# Terminal 1 - Backend
uvicorn elite_craft.api.fastapi_server:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Frontend
streamlit run src/elite_craft/frontend/app.py

# Terminal 3 (Optional) - Proxy
uvicorn elite_craft.api.proxy_server:app --host 0.0.0.0 --port 4000 --reload
```

To stop any service, press `CTRL+C` in its terminal.

### Troubleshooting

**Issue: `ModuleNotFoundError: No module named 'elite_craft'`**

This means you haven't installed the package yet. Run:
```bash
uv pip install -e .
```

This installs the package in editable mode, making the `elite_craft` module available to Python without needing `PYTHONPATH`.

**Issue: Ollama model downloading on first startup**

The FastAPI server will download the required models on first startup:
- Embedding model: `nomic-embed-text:v1.5` (~274MB)
- LLM model: `ministral-3:8b-cloud` (size varies by provider)

This is normal and only happens once. Wait for the downloads to complete before making API requests.

**Issue: `zsh: command not found: pip`**

If you're using `uv` as your package manager, use:
```bash
uv pip install -e .
```

Or use Python's pip module directly:
```bash
python -m pip install -e .
```

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
- `document_id`: Foreign key to documents table
- `chunk_id_in_document`: Chunk sequence number within document
- `content`: Chunk text
- `embedding`: Vector(768) for similarity search
- `created_at`: Timestamp
- Unique constraint on (document_id, chunk_id_in_document)

---

## Testing

Elite Craft maintains comprehensive unit tests for all core services.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov

# Run specific test file
pytest tests/services/test_crawling.py -v

# Run with detailed coverage report
pytest --cov --cov-report=html
```

### Test Coverage

**Core Services:**
- ✅ `config.py` - Configuration and settings management
- ✅ `model_provider.py` - LLM and embedding model providers (Ollama, Groq)
- ✅ `services/crawling.py` - Web crawling and source extraction
- ✅ `services/chunking.py` - Document chunking logic
- ✅ `services/embedding.py` - Embedding generation
- ✅ `services/database_uploading.py` - Database operations
- ✅ `services/update_db_pipeline.py` - End-to-end pipeline orchestration

### Test Structure

```
tests/
├── services/                                # Service layer unit tests
│   ├── test_chunking.py                     # Document chunking tests
│   ├── test_crawling.py                     # Web crawling and source mapping
│   ├── test_database_uploading.py           # Supabase operations
│   ├── test_embedding.py                    # Embedding generation
│   ├── test_model_provider.py               # Model configuration (Ollama, Groq)
│   └── test_update_db_pipeline.py           # Pipeline integration tests
├── test_retrieval_quality.py                # Retrieval quality validation
├── generated_code_verification.py           # Code generation verification
├── are_crawling_outputs_stochastic.py       # Crawl stability analysis script
└── are_db_chunks_and_crawled_chunks_same.py # Database validation script
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

- Built with [LangChain](https://docs.langchain.com/) and [DeepAgents](https://docs.langchain.com/oss/python/deepagents/overview)
- Powered by [Supabase](https://supabase.com/)
- Web search by [Tavily](https://tavily.com/)
- Web crawling by [Crawl4AI](https://docs.crawl4ai.com/)
- Sandboxed execution with [Docker](https://www.docker.com/)