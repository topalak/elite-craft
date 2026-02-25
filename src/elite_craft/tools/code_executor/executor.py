"""
Simple Docker-based code executor with proxy support.
"""

import tempfile
from pathlib import Path
import docker

from config import settings


class CodeExecutor:
    """Execute Python code in Docker sandbox with LLM proxy access."""

    def __init__(self):
        self.client = docker.from_env()
        self.image = "elite-craft/python-sandbox:latest"
        self.proxy_secret = settings.PROXY_SECRET.get_secret_value()

    def execute(self, code: str, timeout: int = 30) -> dict:
        """
        Run Python code in sandbox.

        Args:
            code: Python code to execute
            timeout: Max execution time in seconds

        Returns:
            {
                'stdout': str,
                'stderr': str,
                'exit_code': int,
                'timed_out': bool,
                'execution_time': float
            }
        """
        import time

        container = None
        code_file = None
        start_time = time.time()

        try:
            # Write code to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                code_file = Path(f.name)

            # Create and run container
            container = self.client.containers.create(
                image=self.image,
                command=["python", "-u", "/tmp/code/exec.py"],
                volumes={
                    str(code_file): {'bind': '/tmp/code/exec.py', 'mode': 'ro'}
                },
                environment={
                    "PROXY_SECRET": self.proxy_secret,
                    "LANGSMITH_TRACING": settings.LANGSMITH_TRACING,
                    "LANGSMITH_ENDPOINT": settings.LANGSMITH_ENDPOINT,
                    "LANGSMITH_API_KEY": settings.LANGSMITH_API_KEY.get_secret_value(),
                    "LANGSMITH_PROJECT": "container",
                },
                extra_hosts={'host.docker.internal': 'host-gateway'}, # Access to proxy
                network_mode="bridge",  # Outbound internet access
                user="sandbox",  # Non-root user
                mem_limit="512m",
            )

            container.start()

            # Wait for completion
            try:
                result = container.wait(timeout=timeout)
                exit_code = result['StatusCode']
                timed_out = False
            except:
                container.kill()
                exit_code = -1
                timed_out = True

            # Get output - separate stdout and stderr
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')

            execution_time = time.time() - start_time

            return {
                'stdout': stdout,
                'stderr': stderr,
                'exit_code': exit_code,
                'timed_out': timed_out,
                'execution_time': round(execution_time, 3)
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1,
                'timed_out': False,
                'execution_time': round(execution_time, 3)
            }

        finally:
            # Cleanup
            if container:
                try:
                    container.remove(force=True)
                except:
                    pass
            if code_file and code_file.exists():
                try:
                    code_file.unlink()
                except:
                    pass