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
        "task_type": "wikipedia_search",
        "query": "Build an agent that can search Wikipedia to answer questions",
        "execution_cases": [
            {
                "case_id": "wiki_q1",
                "input": {
                    "input": "Where and when will the 2026 Winter Olympics opening and closing ceremonies take place?"
                },
                "reference_answer": (
                    "The opening ceremony of the 2026 Winter Olympics will take place "
                    "on February 6 at the San Siro stadium in Milan. The closing ceremony "
                    "will be held on February 22 at the Verona Arena."
                )
            },
            {
                "case_id": "wiki_q2",
                "input": {
                    "input": "What new sport will make its Olympic debut at the 2026 Winter Games?"
                },
                "reference_answer": (
                    "Ski mountaineering will make its Olympic debut at the "
                    "2026 Winter Games in Milan-Cortina."
                )
            },
            {
                "case_id": "wiki_q3",
                "input": {
                    "input": "Who are the official mascots of Milano Cortina 2026 and what animals are they based on?"
                },
                "reference_answer": (
                    "The official mascots of Milano Cortina 2026 are Tina and Milo. "
                    "They are based on the stoat (ermine), an animal native to the "
                    "Italian Alps."
                )
            },
            {
                "case_id": "wiki_q4",
                "input": {
                    "input": "How was Milan-Cortina selected as the 2026 Winter Olympics host and who did they beat?"
                },
                "reference_answer": (
                    "Milan-Cortina was selected as the host at the 134th IOC Session "
                    "in June 2019, defeating the rival bid from Stockholm-Åre, Sweden."
                )
            },
            {
                "case_id": "wiki_q5",
                "input": {
                    "input": "What is historically significant about women's participation in the 2026 Winter Olympics?"
                },
                "reference_answer": (
                    "The 2026 Winter Olympics aim to be the most gender-balanced Winter "
                    "Games, with women representing approximately 47% of athletes. "
                    "Women will compete over the same distances as men in cross-country "
                    "skiing for the first time."
                )
            }
        ],
        "scoring": {
            "pass_threshold": 0.8,
            "per_case_weight": 0.2
        }
    }
]

