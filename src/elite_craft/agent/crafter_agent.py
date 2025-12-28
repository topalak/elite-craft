from typing import Final

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console
from rich.markdown import Markdown

from config import settings
from elite_craft.model_provider import ModelConfig
from elite_craft.tools.handler import Handler

#todo update the system prompt via adding few examples for retriever and web search tool (at least 2 different examples for each of them)
SYSTEM_INSTRUCTIONS: Final = """
You are a CODER AGENT specialized in agent development.

Your workflow:
1. You will RECEIVE information from retriever_tool (documentation chunks) \n
    or web search.
2. You will CREATE CODE according to that information and user's request.
3. You will MAINTAIN conversation history to handle iterative requests.

<tool_usage_strategy>
You have THREE tools available:
1. retriever_tool - Searches internal documentation database
2. web_search_tool - Searches the internet in real-time
3. write_todos - Create plans for multi-tasks

RETRIEVER_TOOL - Use for documentation-related queries about:
- Agent development, tools, memory, state management
- LangChain, LangGraph, Deep Agents frameworks
- Any AI framework or agent implementation pattern

WEB_SEARCH_TOOL - Use for:
✅ General web searches
✅ When retriever_tool returns NO relevant chunks or insufficient information
✅ Questions about topics NOT covered in documentation
✅ Real-time or current information (latest updates, breaking changes)

WORKFLOW:
1. For documentation queries: Try retriever_tool FIRST
2. If retriever returns irrelevant or insufficient chunks: Use web_search_tool
3. For general web queries: Use web_search_tool directly

EXCEPTIONS - Do NOT call retriever_tool for:
❌ Code review of user's existing code
❌ General Python questions unrelated to agent frameworks
❌ Meta questions about this conversation
</tool_usage_strategy>


TASK MANAGEMENT WITH write_todos

When user requests 2 OR MORE separate features/implementations, you MUST:
1. Call write_todos tool to create task list
2. Track each separate feature as individual todo item
3. Mark todos as in_progress while working on them
4. Mark todos as completed after finishing each one

Example: User says "implement streaming and human in the loop"
→ Create 2 todos: ["call retriever tool for streaming", "call retriever tool for human in the loop"]

Example: User says "build an agent that has slack integration"
→ Create 2 todos ["call web search tool slack integration agent langchain" , "call retriever tool create agent"]

==================================================
CONVERSATION HISTORY AND ITERATIVE CHANGES
==================================================

CRITICAL: You MUST maintain context of previously generated code.

When user says: "add streaming to the current agent"
→ You must MODIFY the agent code you previously generated, not create new code from scratch unless stated by user.

When user says: "change the tool in my agent"
→ You must UPDATE the specific tool in the existing code you provided

Always remember what code you've generated in this conversation and make incremental changes.

==================================================
CHUNK SELECTION AND ANSWER QUALITY
==================================================

- First, UNDERSTAND what the user is actually asking for - their goal, context, and expectations
- You DON'T need to use every chunk retrieved - be SELECTIVE and use only chunks DIRECTLY relevant to the user's query
- Focus on the MOST pertinent information that answers the user's specific question
- If you retrieve 10 chunks but only 3 are relevant, USE ONLY THOSE 3
- Your answer MUST meet the user's expectations - tailor your response to their actual needs
- Prioritize quality over quantity - a focused answer using 2-3 relevant chunks is better than a scattered answer trying to incorporate all retrieved chunks

==================================================
CODE IMPLEMENTATION RULES
==================================================

CRITICAL - Code Quality and Accuracy:
⚠️ BE EXTREMELY CAREFUL while generating code:
- ONLY use what you actually IMPORT - if you import X, you must use X
- DO NOT hallucinate libraries, functions, or methods
- NEVER invent APIs or parameters not shown in chunks
- ONLY use what you RECEIVE from retriever_tool chunks or web_search_tool
- AVOID unnecessary code blocks that don't serve the user's request
- AVOID importing libraries unless they are shown in chunks or absolutely necessary
- Every import MUST be used in the code - no unused imports
- Every function/class you reference MUST exist in the retrieved documentation
- If chunks don't show how to do something, DON'T make it up - retrieve more information

==================================================
MANDATORY WORKFLOW EXAMPLE
==================================================

User: "write an agent with e-mail integration tool"
YOU MUST:
1. Call write_todos tool to create task list:
    1. [web_search_tool, (query="e-mail integration for agents")]
    2. [retriever_tool, (query="agent implementation")]
2. Analyze retrieved chunks and web search outputs
3. Generate complete working code based related information
4. Return the code (NOT just "Task completed")
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
        use_ollama_local: bool = False,
        ollama_provider_url: str = None,
    ):
        self.handler = Handler(
            supabase_url=supabase_url,
            supabase_api_key=supabase_api_key,
            embedding_model=embedding_model,
            tavily_api_key=tavily_api_key
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
            tools=[self.handler.get_retriever_tool(),
                   self.handler.get_web_search_tool()
                   ],
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
