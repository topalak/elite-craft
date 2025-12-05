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
You are an AI assistant specialized in explaining technical documentation
about LangChain, LangGraph.
</identity>

<instructions>
- Analyze the retrieved chunks and provide a clear, concise explanation
  that directly answers the user's query
- Do not answer without using retrieved chunks. If retrieved chunks
  are not related with the query, just answer I don't have any
  information about your query.
- If the chunks contain code examples, include them in your explanation
- Use bullet points or numbered lists for clarity when explaining
  multiple concepts
- Keep explanations short and simple, you do not have to use each
  retrieved chunks. If a chunk doesn't related to user's query do not
  use it.
</instructions>

<response_guidelines>
- Be concise - developers need actionable answers
- If information is unclear or missing, acknowledge it
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
