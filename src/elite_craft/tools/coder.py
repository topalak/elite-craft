"""Code generator tool with RAG (Retrieval Augmented Generation) pattern."""

from typing import Optional
from pydantic import BaseModel, Field


from elite_craft.model_provider import ModelConfig
from config import settings



class Coder(BaseModel):
    """
    Generates code by synthesizing retrieved cookbook examples.

    This tool implements RAG pattern:
    1. Uses retrieved examples as context (from retriever tool)
    2. Synthesizes/adapts/combines them into user's specific solution
    3. Handles: generation, adaptation, and combination in one tool

    This is NOT generating from scratch - it intelligently applies
    proven patterns from documentation cookbooks.
    """

    query: str = Field(
        description="User's code request. "
                    "Example: 'Create a LangGraph agent with Supabase state persistence'"
    )

    retrieved_examples: list[str] = Field(
        description="Relevant code examples retrieved from cookbooks. "
                    "These provide the context and patterns for generation."
    )

    user_context: Optional[str] = Field(
        default=None,
        description="Additional user-specific context like custom schema, existing code, or requirements"
    )

llm_config = ModelConfig(model='gpt-oss:20b-cloud', use_ollama_cloud=True, api_key=settings.OLLAMA_API_KEY)
llm = llm_config.get_llm()

def generate_code(coder_input: Coder) -> str:
    """
    Generate code using RAG pattern with LLM.

    Args:
        coder_input: Coder model with query and retrieved examples

    Returns:
        Generated Python code as string

    Raises:
        ValueError: If retrieved_examples is empty
        RuntimeError: If LLM generation fails

    Example:
        >>> coder = Coder(
        ...     query="Create LangGraph agent with memory",
        ...     retrieved_examples=["example1", "example2"]
        ... )
        >>> code = generate_code(coder, llm)
    """

    #todo coder_input.context may throw an error if passes empty
    coder_prompt = f"""You are an expert Python developer specializing in LangChain, LangGraph, and Pydantic.

    Your task is to generate production-ready code by synthesizing the provided examples.

    Guidelines:
    - Use async/await where appropriate
    - Add helpful comments with '# Reason:' prefix for complex logic
    ......
    Here is the user's query:
    {coder_input.query}

    Here is retrieved examples:
    {coder_input.retrieved_examples}

    Additional context:
    {coder_input.user_context} 


    DO NOT generate code from scratch - use the provided examples as your foundation."""

    if not coder_input.retrieved_examples:
        raise ValueError("Retrieved examples cannot be empty")


    # Generate code
    try:
        generated_code = llm.invoke([
        {"role":"system", "content": coder_prompt},
            #"query": coder_input.query,
            #"context_section": context_section,
            #"examples": examples_context
        ])
        return generated_code
    except Exception as e:
        raise RuntimeError(f"Code generation failed: {str(e)}") from e


# TODO: Add validation for generated code (syntax check with ast.parse)
# TODO: Add type checking validation with mypy
# TODO: Add caching for common generation patterns
# TODO: Consider adding feedback loop with code_reviewer tool
# TODO: Implement streaming for large code generation