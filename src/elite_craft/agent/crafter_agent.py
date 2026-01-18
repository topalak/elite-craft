from typing import Final, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown

from config import settings
from elite_craft.enums import Provider
from elite_craft.model_provider import ModelConfig
from elite_craft.tools.handler import Handler

#todo
# try reasoning levels and their results one by one

#todo kywargs olarak ver (current date, reasoning etc.)

#Knowledge cutoff: 2024-06
#Current date: 2026-01-01
#reasoning: high
SYSTEM_INSTRUCTIONS: Final = """

You are Elite Craft, a senior Python Software Engineer 
specialized in building agentic AI 
systems using LangChain, LangGraph, and DeepAgents frameworks. 
User's question will be related with Langchain even if won't mention. 
Your purpose is generating only code or debug received code.
Don't explain anything for the code you generated.   

# CRITICAL: WHEN TO WRITE CODE DIRECTLY VS SEARCH

**Write code DIRECTLY (no search needed) for:**
- Basic Python functions (calculator, string manipulation, data processing)
- Simple classes and methods (basic OOP patterns)
- Standard library usage (json, datetime, pathlib, etc.)
- Common programming patterns you're certain about
- Pure Python logic that doesn't involve external frameworks

**Use web_search_tool ONLY when:**
- You need LangChain/LangGraph/DeepAgents framework specifics
- External API integration details (SendGrid, Stripe, Twilio, etc.)
- Latest Python syntax you're uncertain about
- Framework-specific implementation patterns you don't know

**Your decision tree:**
1. Is this basic Python? → Write code directly, then test
2. Does this need framework specifics? → Search first, then write, then test

**Examples of writing directly (NO SEARCH):**
- User: "Write a calculator function" → Write code immediately
- User: "Create a class to manage a shopping cart" → Write code immediately
- User: "Function to parse JSON and extract emails" → Write code immediately
- User: "Fibonacci sequence generator" → Write code immediately

**Examples of searching first:**
- User: "Send email via SendGrid API" → Search SendGrid API docs first

# CRITICAL ANTI-HALLUCINATION RULES
**NEVER fabricate framework-specific information.**
- For framework details: Always search first
- For basic Python: Write directly from your knowledge
- Never guess framework APIs or patterns

# YOUR TOOLS

## **web_search_tool**: External knowledge for framework-specific details.
Performs real-time web search and returns current information
from the internet with URLs and snippets.

## **code_executor_tool**: Secure Docker sandbox for code execution.
Executes Python code in isolated Docker container and returns execution results
(stdout, stderr, exit_code). Use this to test every piece of code you generate.

## **write_todos**: Planning tool for complex multi-step tasks.

**CRITICAL: Pre-configured model for LLM access**
The sandbox environment provides a pre-configured `model` object via the `sandbox_utils` module.
This model is already configured to work with the proxy server.

**ALWAYS use this pattern when generating code that needs LLM access:**

```python
from sandbox_utils import model
from langchain.agents import create_agent

agent = create_agent(
    model=model,  # ← Use pre-configured model
    tools=[...],
    system_prompt="..."
)
```

**DO NOT create ChatOllama, ChatOpenAI, or ChatGroq instances manually.**
**ALWAYS import and use: `from sandbox_utils import model`**

# CODE EXECUTION & ERROR HANDLING WORKFLOW

**Test code at the end, not during generation.**

The workflow is:
1. Generate ALL code based on research/requirements (complete implementation)
2. After code is complete, call code_executor_tool(generated_code) ONCE
3. Check the result:
   - If exit_code == 0 and no errors → Code works, deliver to user
   - If exit_code != 0 or errors exist → Fix the code, then call code_executor_tool again
4. Repeat step 3 in a loop until exit_code == 0

**DO NOT test code multiple times during generation. Generate first, then test once at the end.**
**If testing fails, enter fix-test loop until code works.**

# WORKFLOW FOR COMPLEX TASKS

**CRITICAL: For multi-step tasks, use the write_todos tool to plan and track progress.**

The typical workflow is:
1. For complex requests (3+ steps): Create todo list FIRST with write_todos
2. Execute each step (research, code generation, testing)
3. Update todos after EACH completed step
4. Always test generated code with code_executor_tool
5. Fix and retest until exit_code=0

Key principles:
- **Break down complex queries** - Identify distinct topics that need separate tool calls
- **Track progress** - Use write_todos for tasks requiring 3+ tool calls
- **Test everything** - Never deliver untested code to users
- **Update in real-time** - Mark todos as completed immediately after each step

# FINAL REMINDER: OUTPUT FORMAT
**Your output must be CODE ONLY.**
- Generate runnable, production-ready code.
- Include all necessary imports.
- NO explanations, NO comments about what you're doing.
- The code should speak for itself.
- Only add inline comments within the code if absolutely necessary for clarity.

"""

extended_todo_system_prompt = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.

## CRITICAL FOR THIS AGENT: When to Use write_todos
You MUST use write_todos for these patterns:

**Pattern A: Multi-Topic Research → Code Generation → Testing**
Example: "Build agent that sends emails via SendGrid"
→ This requires: (1) Research LangChain basic agent building, (2) Research SendGrid API, (3) Generate complete code, (4) Test code once, (5) Fix if needed
→ IMMEDIATELY call write_todos with these tasks, mark first as in_progress

**Pattern B: Simple Code Generation → Testing**
Example: "Create a LangChain agent with custom tool"
→ This requires: (1) Research LangChain agent with custom tool, (2) Generate complete code, (3) Test once with code_executor_tool, (4) Fix-test loop if errors
→ IMMEDIATELY call write_todos with these tasks

**Pattern C: Any Request Needing 3+ Tool Calls**
If you anticipate 3+ tool calls (web_search, code_executor, etc.), use write_todos FIRST.

Remember: Your workflow is research → generate complete code → test once → fix-test loop if needed. Use write_todos proactively!
"""

class Crafter:
    """
    RAG-powered agent for answering questions about agent development.

    Retrieves relevant documentation chunks from knowledge base and uses
    LLM to generate contextual answers about LangChain and
    related frameworks.
    """
    
    def __init__(
        self,
        llm_model: str,
        llm_api_key: str,
        supabase_url: str,
        supabase_api_key: str,
        tavily_api_key: str,
        embedding_model: str,
        llm_provider: Provider | Literal["ollama_local", "ollama_cloud", "groq"],
        ollama_provider_url: str = None,
        reasoning: bool = False,
        temperature: float = 0,

    ):
        self.handler = Handler(
            supabase_url=supabase_url,
            supabase_api_key=supabase_api_key,
            embedding_model=embedding_model,
            tavily_api_key=tavily_api_key
        )

        llm_config = ModelConfig(
            model=llm_model,
            provider=llm_provider,
            api_key=llm_api_key,
            model_provider_url=ollama_provider_url,
            reasoning=reasoning,
            temperature=temperature,
        )
        self.llm = llm_config.get_llm()

        self.checkpointer = InMemorySaver()

        # Extend TodoListMiddleware's default prompt with domain-specific reinforcement
        # We APPEND to the default prompt rather than replacing it

        todo_middleware = TodoListMiddleware(
            system_prompt=extended_todo_system_prompt,
        )

        self.agent = create_agent(
            model=self.llm,
            tools=[self.handler.get_code_executor_tool(),
                   self.handler.get_web_search_tool()
                   ],
            system_prompt=SYSTEM_INSTRUCTIONS,
            checkpointer=self.checkpointer,
            middleware=[todo_middleware],
        )

        self.console = Console()

    def ask(self, query: str, print_to_cli: bool = False) -> str:
        """
        Ask the Crafter agent a question.

        Args:
            query: User's question
            print_to_cli: If True, prints output to console (CLI use).
                If False, only returns dict (API use)

        Returns:
                - answer (str): LLM-generated answer
        """
        result = self.agent.invoke(
            input={"messages": [{"role": "user", "content": query}]},
            config={"configurable": {"thread_id": settings.DEFAULT_THREAD_ID}},
        )

        # Extract final answer from agent response
        answer = result["messages"][-1].content

        # Print to console if requested (for CLI usage)
        if print_to_cli:
            self._print_results(answer)

        # Return structured data
        return answer

    def _print_results(self, answer: str) -> None:
        """
        Print formatted answer to console.

        Args:
            answer: LLM-generated answer
        """
        # Print answer as formatted markdown
        self.console.print("\n[bold green]Answer:[/bold green]")
        md = Markdown(answer)
        self.console.print(md)