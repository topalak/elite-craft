# Elite Craft

**Forging Elite AI Systems**

An AI agent framework designed to help developers build and enhance agentic AI projects. 

---
## Project Structure

```
elite-craft/
├── src/
│   ├── config.py                    # Configuration management
│   └── elite_craft/
│       ├── __init__.py
│       ├── services/                # Core services
│       │   ├── crawling.py          # Web crawling
│       │   ├── chunking.py          # Document chunking
│       │   ├── embedding.py         # Text embeddings
│       │   ├── database_uploading.py # Database operations
│       │   └── update_db_pipeline.py # Orchestration pipeline
│       ├── database/
│       │   └── db_setup.sql         # Database schema
│       ├── tools/                   # LangChain tools
│       ├── agent/                   # Agent implementation
│       └── model_provider.py        # LLM configuration
├── CLAUDE.md                        # Coding standards
└── README.md
```
## Overview

Elite Craft is an intelligent agent that generates code by providing:
- Documentation crawling and knowledge base management
- Semantic search across framework documentation
- Code generation and review capabilities
- Best practices guidance for agent development

---

## Architecture

### Core Services

- **Crawler**: Asynchronous web crawling service using Crawl4AI
- **Chunker**: Document chunking using Docling's hybrid strategy
- **Embedder**: Text embedding generation with Ollama models
- **Database Uploader**: Supabase integration for metadata and vector storage
- **Pipeline**: End-to-end processing orchestration

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
git clone https://github.com/yourusername/elite-craft.git
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

Create a `.env` file in the project root:

```env
# Ollama
OLLAMA_HOST_LOCAL=http://localhost:11434

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_SECRET_KEY=your-service-role-key

# LangSmith (optional)
LANGSMITH_API_KEY=your-api-key
LANGSMITH_TRACING=true
```

---

## Usage

### Running the Update Pipeline

Process documentation URLs and build your knowledge base:

```python
import asyncio
from elite_craft.services import UpdateDBPipeline

async def main():
    pipeline = UpdateDBPipeline()

    urls = [
        "https://docs.langchain.com/oss/python/langgraph/overview",
        "https://docs.langchain.com/oss/python/langgraph/tutorials/introduction",
    ]

    results = await pipeline.process(urls)
    print(f"Processed {len(results)} URLs")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Development

### Coding Standards

This project follows strict Python coding standards defined in `CLAUDE.md`:

- **KISS Principle**: Keep it simple, stupid
- **YAGNI**: You aren't gonna need it
- **Fail Fast**: Check for errors early
- Google-style docstrings
- Type hints throughout
- Comprehensive error handling
- Structured logging

### Running Tests

```bash
# Run tests (when available)
pytest tests/
```

---

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

## Contributing

Contributions are welcome! Please ensure:

1. Code follows `CLAUDE.md` standards
2. All functions have proper docstrings
3. Error handling is comprehensive
4. Logging is used (not print statements)
5. Type hints are included

---

## Acknowledgments

- Built with [LangChain](https://docs.langchain.com/)
- Powered by [Supabase](https://supabase.com/)
- Document processing by [Docling](https://docling-project.github.io/)
- Web crawling by [Crawl4AI](https://docs.crawl4ai.com/)