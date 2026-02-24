from src.elite_craft.model_provider import ModelConfig
from pydantic import SecretStr, BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional

PROMPT = """
You are an expert evaluator assessing the quality of an AI-generated response.
Your judgment must be rigorous, consistent, and grounded in evidence.

## Context
- **Task Type**:
 {task_type}
- **User Query**:
 {query}
- **Generated Response**:
 {response}
- **Reference Answer**:
 {reference_answer}

## Evaluation Dimensions

Assess the response on each dimension using a score from 1–5,
where 1 = poor and 5 = excellent.

### 1. Answer Relevance
Does the response directly and completely address the user's query?
Penalize responses that drift, pad, or partially answer.

### 2. Factual Accuracy
Based on your knowledge, are the claims in the response accurate?
Flag any statements that are likely hallucinated, misleading, or unverifiable.

### 3. Completeness
Does the response cover all aspects of the query?
Note any important omissions or unexplored angles.

### 4. Coherence & Clarity
Is the response logically structured, well-reasoned,
and easy to understand?

### 5. Conciseness
Does the response avoid unnecessary verbosity, repetition,
or filler content while still being thorough?

## Hard Rules (Override all scores)
- If the response does not address the query at all → answer_relevance = 1,
- If the response contains a clear factual error → factual_accuracy = 1, 
- If a reference answer is provided and the response directly contradicts it
  → factual_accuracy = 1,
"""


class EvaluationResult(BaseModel):
    answer_relevance: int = Field(description="Answer relevance",gt=0, lt=6)
    factual_accuracy: int = Field(description="Factual accuracy",gt=0, lt=6)
    completeness: int = Field(description="Completeness",gt=0, lt=6)
    conciseness: int = Field(description="Conciseness",gt=0, lt=6)
    supported_claims: List[str] = Field(description="Each item must be a plain string describing one accurate, well-reasoned claim")
    disputed_claims: List[str] = Field(description="Each item must be a plain string describing one inaccurate, vague, or unverifiable claim")



class Evaluator:

    def __init__(self, model:str, provider:str, api_key:SecretStr):
        llm_config = ModelConfig(model=model,
                                 provider=provider,
                                 api_key=api_key.get_secret_value())

        llm = llm_config.get_llm()

        self.structured_model = llm.with_structured_output(
            schema=EvaluationResult,
            include_raw=True,
            method='function_calling'
        )

    def evaluate(self,
                 task_type:str,
                 query:str ,
                 response:str,
                 reference_answer:str
                 ):

        instructions = PROMPT.format(
            task_type=task_type,
            query=query,
            response=response,
            reference_answer=reference_answer
        )

        response = self.structured_model.invoke(instructions)
        result: EvaluationResult = response["parsed"]

        return self._calculate_weighted_score(result)

    def _calculate_weighted_score(self, result: EvaluationResult) -> float:
        """Calculate weighted evaluation score.

        Args:
            result: Parsed evaluation result from the LLM

        Returns:
            Weighted score between 0.9 and 4.5
        """
        return (
            result.answer_relevance * 0.35
            + result.factual_accuracy * 0.30
            + result.completeness * 0.15
            + result.conciseness * 0.10
        )







