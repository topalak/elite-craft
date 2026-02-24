EVAL_DATASET = [
    # Easy: Basic agent
    {
        "task_id": "agent_basic_001",
        "query": "Create a simple agent that takes a topic and generates a one-sentence summary",
        "execution": {
            "input": {"topic": "photosynthesis"},
        },
        "verification": {
            "type": "llm_judge",
            "criteria": "Output is a single coherent sentence about photosynthesis"
        }
    },

    # Easy: Structured output
    {
        "task_id": "structured_001",
        "query": "Use LangChain to extract name and age from text into a structured format",
        "execution": {
            "input": {"text": "Maria is 32 years old"},
        },
        "verification": {
            "type": "custom",
            "validator": lambda out: {
                "passed": "maria" in str(out).lower() and "32" in str(out),
                "reason": "Should extract Maria and 32"
            }
        }
    },

    # Medium: Tool-calling agent with math
    {
        "task_id": "agent_math_001",
        "query": "Create a LangChain agent with a calculator tool that can solve math problems",
        "execution": {
            "input": {"input": "What is 145 * 12?"},
        },
        "verification": {
            "type": "contains",
            "expected": "1740"
        }
    },

    # Medium: Wikipedia tool
    {
        "task_id": "agent_wiki_001",
        "query": "Build an agent that can search Wikipedia to answer questions",
        "execution": {
            "input": {"input": "What year was the Eiffel Tower completed?"},
        },
        "verification": {
            "type": "contains",
            "expected": "1889"
        }
    },
]

