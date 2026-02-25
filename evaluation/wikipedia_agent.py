#############################################################################

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain.agents import create_agent


# Create Wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# Create agent with Wikipedia tool
agent = create_agent(
    model=model,
    tools=[wikipedia_tool]
)
query = "Where and when will the 2026 Winter Olympics opening and closing ceremonies take place?"
reference_answer = """The opening ceremony will be held at Stadio San Siro in Milan on February 6, 2026, themed 'Armonia' (Harmony). The closing ceremony will be held at the Verona Arena on February 22, 2026, themed "Beauty in Action."""
# Test the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": query}]
})
evaluator.evaluate(task_type="Question Answering",query=query, response=result['messages'][-1].content, reference_answer=reference_answer)