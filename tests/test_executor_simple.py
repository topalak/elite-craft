"""Test Docker executor."""

import sys
sys.path.insert(0, 'src')

from elite_craft.tools.code_executor import CodeExecutor

executor = CodeExecutor()

# Test 1: Simple print
print("=== Test 1: Simple print ===")
result = executor.execute("print('Hello from sandbox')")
print(f"Exit code: {result['exit_code']}")
print(f"Timed out: {result['timed_out']}")
print(f"Output: {result['stdout']}")

# Test 2: Error
print("\n=== Test 2: Error ===")
result = executor.execute("raise ValueError('Test error')")
print(f"Exit code: {result['exit_code']}")
print(f"Output:\n{result['stdout']}")

# Test 3: LangChain
print("\n=== Test 3: LangChain ===")
result = executor.execute("import langchain; print(f'LangChain {langchain.__version__}')")
print(f"Exit code: {result['exit_code']}")
print(f"Output: {result['stdout']}")