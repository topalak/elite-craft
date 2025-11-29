from typing import Final

from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, BaseMessage
from rich.console import Console
from rich.markdown import Markdown

from elite_craft.model_provider import ModelConfig
from elite_craft.tools.retriever import Retriever

SYSTEM_INSTRUCTIONS: Final = """
<identity>
You are an AI assistant specialized in explaining technical documentation about LangChain, LangGraph.
</identity>

<instructions>
- Analyze the retrieved chunks and provide a clear, concise explanation that directly answers the user's query
- If the chunks contain code examples, include them in your explanation
- Use bullet points or numbered lists for clarity when explaining multiple concepts
- Keep explanations short and simple, you do not have to use each retrieved chunks. 
If a chunk doesn't related to user's query do not use it. 
</instructions>

<response_guidelines>
- Be concise - developers need actionable answers
- If information is unclear or missing, acknowledge it
</response_guidelines>
"""
FORMATTED_TEXT : Final = """ <task>
The user asked: "{query}"

Below are relevant documentation chunks retrieved from the knowledge base:

<retrieved_chunks>
{chunks}
</retrieved_chunks>
</task>

"""
class Memory(BaseModel):
    context_window : list[BaseMessage]

class Crafter:

    def __init__(self,
                 llm_model:str,
                 llm_api_key:str,
                 embedding_model_name: str,
                 supabase_url: str,
                 supabase_api_key: str
                 ):

        self.retriever = Retriever(supabase_url=supabase_url,
                                   supabase_api_key=supabase_api_key,
                                   embedding_model_name=embedding_model_name)

        llm_config = ModelConfig(model=llm_model, api_key=llm_api_key)
        self.llm = llm_config.get_llm()
        self.state: Memory = Memory(
            context_window=[SystemMessage(content=SYSTEM_INSTRUCTIONS)],
        )
        self.console = Console()

    def ask(self, query: str) -> None:

        self.state.context_window.append(HumanMessage(content=query))

        response = self.retriever.retrieve_relevant_chunks(query)

        content_list = [chunk['content'] for chunk in response]

        # Format chunks with separators
        chunks_text = "\n\n---\n\n".join(content_list)

        # Format the system prompt with query and chunks
        formatted_prompt = FORMATTED_TEXT.format(query=query, chunks=chunks_text)

        self.state.context_window.append(ToolMessage(content=formatted_prompt, tool_call_id="retriever"))

        # Get LLM response
        llm_response = self.llm.invoke(self.state.context_window)

        self.state.context_window.append(llm_response)

        retrieved_chunks = [
            (f"Document Url: {chunk['url']} \n"
             f"Document Id: {chunk['chunk_id_in_document']}  \n"
             f"Content Preview: {chunk['content'][:50]}")
            for chunk in response ]

        # Print retrieved chunks
        self.console.print("\n[bold cyan]Retrieved Chunks:[/bold cyan]")
        for idx, chunk_info in enumerate(retrieved_chunks, 1):
            self.console.print(f"[yellow]{idx}.[/yellow] {chunk_info}\n")

        md = Markdown(llm_response.content)

        self.console.print(md)

        return None

