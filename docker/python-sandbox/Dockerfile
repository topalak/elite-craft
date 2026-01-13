# Elite Craft - Sandboxed Python Code Executor
# Lightweight, secure container for running generated code

FROM python:3.13-slim

# Security: Create non-root user
RUN useradd -m -u 1000 sandbox && \
    mkdir -p /sandbox /tmp/code && \
    chown -R sandbox:sandbox /sandbox /tmp/code

# Install common dependencies your agents might generate code for
# Pre-installing avoids runtime pip installs (faster + more secure)
RUN pip install --no-cache-dir \
    langchain>=1.1.0 \
    langchain-core>=1.2.7 \
    pydantic>=2.10.5 \
    numpy>=2.2.1 \
    pandas>=2.2.3 \
    requests>=2.32.3

# Set working directory
WORKDIR /sandbox

# Switch to non-root user for execution
USER sandbox

# Default command: execute code from stdin
CMD ["python", "-u", "/tmp/code/exec.py"]
