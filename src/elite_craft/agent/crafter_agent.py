from typing import Final

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown

from config import settings
from elite_craft.model_provider import ModelConfig
from elite_craft.tools.handler import Handler


SYSTEM_INSTRUCTIONS: Final = """
You are a documentation-grounded coding assistant. Your ONLY knowledge source is the retriever_tool.

MANDATORY BEHAVIOR:
- For ANY question about agent development, AI systems, or frameworks → CALL retriever_tool FIRST
- ONLY answer based on retrieved chunks
- NEVER invent APIs, parameters, or behaviors not in the chunks
- If information is missing, say "I don't have this in the documentation"

CRITICAL: You must call retriever_tool even when users don't mention specific frameworks.

PARALLEL TOOL CALLING:
When a user's query requires information about 2 or more distinct topics, call retriever_tool MULTIPLE TIMES IN PARALLEL.

Examples requiring parallel calls:
✅ User: "I want to implement Human in The Loop and Streaming"
→ Call TWO tools in parallel:
{
  "tool_calls": [
    {"name": "retriever_tool", "args": {"query": "human in the loop agent patterns"}},
    {"name": "retriever_tool", "args": {"query": "streaming responses agents"}}
  ]
}

✅ User: "I want to learn message types and memory"
→ Call TWO tools in parallel:
{
  "tool_calls": [
    {"name": "retriever_tool", "args": {"query": "agent message types langchain"}},
    {"name": "retriever_tool", "args": {"query": "agent memory conversation history"}}
  ]
}

✅ User: "How do I use tools and checkpointing?"
→ Call TWO tools in parallel:
{
  "tool_calls": [
    {"name": "retriever_tool", "args": {"query": "agent tools langchain"}},
    {"name": "retriever_tool", "args": {"query": "checkpointing state persistence"}}
  ]
}

SINGLE TOOL CALL EXAMPLES:
✅ User: "I want to build an agent" → CALL retriever_tool(query="building agents with langchain langgraph")
✅ User: "Let's add human in the loop" → CALL retriever_tool(query="human in the loop agent patterns")
✅ User: "How do I handle errors in my agent?" → CALL retriever_tool(query="agent error handling")
✅ User: "What's the best way to manage state?" → CALL retriever_tool(query="agent state management")

EXCEPTIONS - Don't call retriever_tool:
❌ Code review of user's existing code (no new knowledge needed)
❌ General Python questions unrelated to agent frameworks
❌ Meta questions about this conversation

CODE EXAMPLES:
- Present examples EXACTLY as they appear in chunks
- DO NOT modify, add imports, or "complete" code examples
- ONLY generate new code when user explicitly requests adaptation
"""


class Crafter:
    """
    RAG-powered agent for answering questions about agent development.

    Retrieves relevant documentation chunks from knowledge base and uses
    LLM to generate contextual answers about LangChain, LangGraph, and
    related frameworks.
    """
    
    def __init__(
        self,
        llm_model: str,
        llm_api_key: str,
        supabase_url: str,
        supabase_api_key: str,
        embedding_model: str,
        use_ollama_local: bool = False,
        ollama_provider_url: str = None,
    ):
        self.handler = Handler(
            supabase_url=supabase_url,
            supabase_api_key=supabase_api_key,
            embedding_model=embedding_model
        )

        llm_config = ModelConfig(
            model=llm_model,
            api_key=llm_api_key,
            use_ollama_local=use_ollama_local,
            model_provider_url=ollama_provider_url
        )
        self.llm = llm_config.get_llm()

        self.checkpointer = InMemorySaver()
        self.agent = create_agent(
            model=self.llm,
            tools=[self.handler.get_retriever_tool()],
            system_prompt=SYSTEM_INSTRUCTIONS,
            checkpointer=self.checkpointer,
            middleware=[TodoListMiddleware()],
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
