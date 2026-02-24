# Python Sandbox Docker Image

Secure, isolated environment for executing agent-generated Python code.

## Build Image

```bash
docker build -t elite-craft/python-sandbox:latest ./docker/python-sandbox
```

## Security Features

- Non-root user execution (UID 1000)
- Slim base image (minimal attack surface)
- Pre-installed dependencies (no runtime pip install)
- Resource limits enforced by Docker client

## Pre-installed Packages

- langchain, langchain-core, langgraph
- pydantic
- numpy, pandas
- requests

## Manual Testing

```bash
echo 'print("Hello from sandbox")' > test.py
docker run --rm -v $(pwd)/test.py:/tmp/code/exec.py elite-craft/python_sandbox:latest
```

## Adding New Dependencies

Edit `Dockerfile` and rebuild. Keep the image lean - only add packages your agents commonly generate code for.