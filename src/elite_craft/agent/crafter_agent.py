from typing import Final

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown

from elite_craft.model_provider import ModelConfig
from elite_craft.tools.retriever import Retriever


SYSTEM_INSTRUCTIONS: Final = """
<identity>
You are an elite AI coding agent specialized in LangChain and LangGraph
development. Your primary role is to help developers build, debug, and
optimize agent-based applications using retrieved documentation as your
knowledge source.
</identity>

<core_capabilities>
- Code Generation: Write production-ready code with proper error handling
</core_capabilities>

<instructions>
1. Use retrieved documentation chunks as your primary knowledge source if
    they satisfy with user's query, if it isn't DON'T use unrelated ones.
2. If retrieved chunks are not relevant to the query, respond with:
   "I don't have relevant documentation for this query in my knowledge base."
3. When generating code:
   - Be concise while generating code as possible as you can
   - Follow Python best practices (PEP 8, type hints, docstrings)
   - Include proper error handling and validation
   - Add inline comments for complex logic with # Reason: prefix
   - Use async/await patterns where appropriate
4. Structure your responses based on query type:
   - "How do I...?" → Provide working code example + explanation
   - "What is...?" → Explain concept + minimal code snippet
   - "Debug this..." → Analyze issue + corrected code
   - "Best practice for..." → Recommend pattern + implementation
5. Include code examples from retrieved chunks when available
6. Reference specific documentation sources for key points
</instructions>

<code_quality_standards>
- Use Google-style docstrings for functions and classes
- Add type hints for all function parameters and returns
- Keep functions focused and single-purpose
- Use descriptive variable names (snake_case)
- Maximum line length: 100 characters
- Prefer composition over inheritance
- Use Pydantic models for data validation
</code_quality_standards>

<response_format>
For coding queries, structure responses as:

1. **Quick Answer**: One-sentence summary of the solution
2. **Code Implementation**: Complete, runnable code example
3. **Explanation**: Key points about how/why it works
4. **Important Notes**: Edge cases, gotchas, or best practices
5. **Related Concepts**: Links to related documentation (if relevant)

For conceptual queries:
- Brief definition
- When to use it
- Simple code example
- Key considerations
</response_format>

<error_handling>
When you encounter:
- Incomplete documentation: Acknowledge gaps and provide best-effort solution
- Ambiguous queries: Ask clarifying questions before generating code
- Multiple valid approaches: Present the most common/recommended pattern
- Outdated patterns in chunks: Note if documentation seems outdated
</error_handling>

<response_guidelines>
- Be concise but complete - provide working solutions, not pseudo-code
- Prioritize correctness over cleverness
- Include necessary imports and dependencies
- Test-aware: Mention how to test the solution when relevant
- Security-conscious: Flag potential security issues
- Performance-aware: Note performance implications for critical code
- Always cite documentation chunks used (e.g., "According to the docs...")
</response_guidelines>
"""
FORMATTED_TEXT: Final = """ <task>
The user asked: "{query}"

Below are relevant documentation chunks retrieved from the knowledge base:

<retrieved_chunks>
{chunks}
</retrieved_chunks>
</task>

"""


class Memory(BaseModel):
    """
    Agent memory for maintaining conversation context.

    Attributes:
        context_window: List of messages in conversation history
    """

    context_window: list[BaseMessage]


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
        embedding_model_name: str,
        supabase_url: str,
        supabase_api_key: str
    ):

        self.retriever = Retriever(
            supabase_url=supabase_url,
            supabase_api_key=supabase_api_key,
            embedding_model_name=embedding_model_name
        )

        llm_config = ModelConfig(model=llm_model, api_key=llm_api_key)
        self.llm = llm_config.get_llm()
        self.state: Memory = Memory(
            context_window=[SystemMessage(content=SYSTEM_INSTRUCTIONS)],
        )
        self.console = Console()

    def ask(self, query: str, print_to_cli: bool = False) -> dict:
        """
        Ask the Crafter agent a question.

        Args:
            query: User's question
            print_to_cli: If True, prints output to console (CLI use).
                If False, only returns dict (API use)

        Returns:
            Dict with keys:
                - answer (str): LLM-generated answer
                - chunks (list[dict]): Retrieved documentation chunks
        """
        self.state.context_window.append(HumanMessage(content=query))

        response = self.retriever.retrieve_relevant_chunks(query)

        content_list = [chunk['content'] for chunk in response]

        # Format chunks with separators
        chunks_text = "\n\n---\n\n".join(content_list)

        # Format the system prompt with query and chunks
        formatted_prompt = FORMATTED_TEXT.format(
            query=query,
            chunks=chunks_text
        )

        self.state.context_window.append(
            ToolMessage(
                content=formatted_prompt,
                tool_call_id="retriever"
            )
        )

        # Get LLM response
        llm_response = self.llm.invoke(self.state.context_window)

        self.state.context_window.append(llm_response)

        # Print to console if requested (for CLI usage)
        if print_to_cli:
            retrieved_chunks = [
                (f"Document Url: {chunk['url']} \n"
                 f"Document Id: {chunk['chunk_id_in_document']}  \n"
                 f"Content Preview: {chunk['content'][:50]}")
                for chunk in response
            ]

            # Print retrieved chunks
            self.console.print(
                "\n[bold cyan]Retrieved Chunks:[/bold cyan]"
            )
            for idx, chunk_info in enumerate(retrieved_chunks, 1):
                self.console.print(
                    f"[yellow]{idx}.[/yellow] {chunk_info}\n"
                )

            md = Markdown(llm_response.content)
            self.console.print(md)

        # Return structured data
        return {
            "answer": llm_response.content,
            "chunks": response
        }
