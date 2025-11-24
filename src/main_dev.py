from typing import Final

from elite_craft.model_provider import ModelConfig
from elite_craft.agent.crafter_agent import Crafter
from src.config import settings

SYSTEM_PROMPT: Final = """
<identity>
You are an AI assistant specialized in explaining technical documentation about LangChain, LangGraph, Pydantic, and Supabase.
</identity>

<task>
The user asked: "{query}"

Below are relevant documentation chunks retrieved from the knowledge base:

<retrieved_chunks>
{chunks}
</retrieved_chunks>
</task>

<instructions>
- Analyze the retrieved chunks and provide a clear, concise explanation that directly answers the user's query
- If the chunks contain code examples, include them in your explanation
- Use bullet points or numbered lists for clarity when explaining multiple concepts
- Keep explanations practical and developer-focused
</instructions>

<response_guidelines>
- Be concise - developers need actionable answers
- If information is unclear or missing, acknowledge it
</response_guidelines>
"""

if __name__ == "__main__":
    llm_config = ModelConfig(model=settings.LLM_NAME, api_key=settings.OLLAMA_API_KEY)
    llm = llm_config.get_llm()
    crafter = Crafter(
                      embedding_model_name=settings.EMBEDDING_MODEL, supabase_url=settings.SUPABASE_URL,
                      supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY)

    while True:
        query = input(">You: ")

        if query.lower() == 'exit':
            break

        # Get retrieved chunks
        chunks_data = crafter.ask(query)
        content_list = [chunk['content'] for chunk in chunks_data]

        # Format chunks with separators
        chunks_text = "\n\n---\n\n".join(content_list)

        # Format the system prompt with query and chunks
        formatted_prompt = SYSTEM_PROMPT.format(query=query, chunks=chunks_text)

        # Get LLM response
        llm_response = llm.invoke(formatted_prompt)

        print(f"\nAssistant: {llm_response.content}\n")
