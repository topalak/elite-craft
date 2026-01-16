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
        self.proxy_secret = settings.PROXY_SECRET

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
                'timed_out': bool
            }
        """
        container = None
        code_file = None

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
                environment={"PROXY_SECRET": self.proxy_secret},
                extra_hosts={'host.docker.internal': 'host-gateway'}, # Access to proxy
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

            # Get output
            logs = container.logs(stdout=True, stderr=True).decode('utf-8')

            return {
                'stdout': logs,
                'stderr': '',
                'exit_code': exit_code,
                'timed_out': timed_out
            }

        except Exception as e:
            return {
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1,
                'timed_out': False
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