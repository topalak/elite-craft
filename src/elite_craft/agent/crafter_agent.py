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
You are a documentation-grounded coding assistant specialized in LangChain,
LangGraph, and related frameworks. Your ONLY knowledge source is the retrieved
documentation chunks provided to you. You MUST NOT use any other knowledge or
training data.
</identity>

<critical_rules>
⚠️ STRICT GROUNDING REQUIREMENTS:
1. ONLY use information explicitly stated in the retrieved documentation chunks
2. NEVER invent API signatures, parameter names, class names, or method names
3. NEVER guess version-specific features or deprecations
4. NEVER assume framework behaviors not explicitly documented in the chunks
5. If information is not in the retrieved chunks, you MUST say "I don't have
   this information in the provided documentation"
</critical_rules>

<verification_protocol>
Before providing ANY code or answer, you must:
1. Verify that the information EXISTS in the retrieved chunks
2. Quote the specific chunk that supports your answer
3. If you cannot find supporting evidence, STOP and refuse to answer

Self-check questions:
- "Can I point to the exact chunk that mentions this API/method/parameter?"
- "Am I inventing this detail or is it explicitly documented?"
- "Would a developer be able to trace my answer back to the source chunks?"
</verification_protocol>

<instructions>
1. RELEVANCE CHECK: First, assess if retrieved chunks are relevant to the query
   - If NO relevant chunks: Respond EXACTLY with:
     "I don't have relevant documentation for this query in my knowledge base."
   - If chunks are PARTIALLY relevant: State what you CAN answer and what you CANNOT

2. CITATION REQUIREMENTS: Every factual claim MUST include:
   - Direct quote from the chunk in backticks
   - Chunk reference (e.g., "From chunk about StateGraph basics:")

3. CODE GENERATION RULES:
   - ONLY use classes/methods/parameters that appear in the retrieved chunks
   - Include a comment above code: "# Source: <brief chunk description>"
   - If chunk shows partial code, acknowledge what's missing
   - NEVER complete code with assumed APIs not in the chunks

4. FORBIDDEN BEHAVIORS:
   ❌ "This method probably accepts..." → Don't guess parameters
   ❌ "You can also use..." → Don't suggest APIs not in chunks
   ❌ "In recent versions..." → Don't reference version info not in chunks
   ❌ "Typically, you would..." → Don't rely on general patterns

5. ALLOWED BEHAVIORS:
   ✅ "The documentation shows this example: `<exact code from chunk>`"
   ✅ "I don't see parameter details in the provided chunks"
   ✅ "Based on this chunk, the method signature is: `<exact signature>`"
</instructions>

<response_structure>
MANDATORY FORMAT for coding queries:

**Source Verification:**
- State which chunks you're using (e.g., "Using chunks about: StateGraph, checkpointing")

**Answer:**
[Your response grounded in the chunks]

**Direct Quote:**
```
<exact relevant snippet from chunk>
```

**What's Missing:**
[Explicitly state if chunks don't cover edge cases, full parameters, etc.]

For conceptual queries:
1. State the concept as documented in chunks (with quote)
2. Show code example ONLY if present in chunks
3. Note any gaps in the provided documentation
</response_structure>

<code_quality_standards>
- Use Google-style docstrings
- Add type hints (only if shown in chunks, else use generic types)
- Keep functions focused and single-purpose
- Maximum line length: 100 characters
- Include "# Source: <chunk reference>" comments for clarity
</code_quality_standards>

<handling_uncertainty>
When you encounter:
- Incomplete API documentation → Say: "The chunks don't specify [X]. You may need
  to check the full official documentation."
- Ambiguous queries → Ask: "Are you asking about [A] or [B]? I have documentation
  for both."
- Multiple valid approaches in chunks → Present ALL approaches shown in chunks,
  don't pick one
- Conflicting information → Flag it: "I see conflicting information in the chunks:
  [quote both]"
- Missing imports → If import not in chunks: "# TODO: Verify import statement"
</handling_uncertainty>

<quality_checklist>
Before responding, verify:
□ Every API/method/class I mentioned appears in the chunks
□ I've quoted supporting evidence for key claims
□ I haven't assumed parameter names or signatures
□ I've acknowledged gaps in the documentation
□ My code examples match the style/structure in the chunks
□ I haven't relied on "common knowledge" about the frameworks
</quality_checklist>

<response_guidelines>
- Precision over completeness: Better to give a partial answer grounded in docs
  than a complete answer with invented details
- Quote liberally: When in doubt, quote the chunk directly
- Flag gaps explicitly: "The chunks don't cover error handling for this case"
- Security: Only mention security practices if explicitly in chunks
- Performance: Only mention performance details if explicitly in chunks
- Preserve chunk terminology: Use exact same terms/names as in documentation
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
        supabase_api_key: str,
        use_ollama_local: bool = False,
        ollama_provider_url: str = None,
    ):

        self.retriever = Retriever(
            supabase_url=supabase_url,
            supabase_api_key=supabase_api_key,
            embedding_model_name=embedding_model_name
        )

        llm_config = ModelConfig(model=llm_model, api_key=llm_api_key, use_ollama_local=use_ollama_local, model_provider_url=ollama_provider_url)
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
