"""
Evaluation pipeline for Elite Craft agent.

Orchestrates end-to-end evaluation: generates agent code via Crafter,
executes it with test inputs in Docker sandbox, and scores outputs
with an LLM-as-judge evaluator.
"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime

from config import settings
from elite_craft.agent.crafter_agent import Crafter
from elite_craft.enums import Model, Provider
from elite_craft.tools import CodeExecutor
from evaluator import Evaluator
from evaluation_dataset import EVAL_DATASET

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SANDBOX_TIMEOUT = 120
MAX_SCORE = 4.5


def _create_crafter() -> Crafter:
    """Create a Crafter instance from application settings."""
    return Crafter(
        llm_model=settings.LLM_NAME,
        llm_api_key=settings.OLLAMA_API_KEY.get_secret_value(),
        llm_provider=Provider(settings.LLM_PROVIDER),
        ollama_provider_url=settings.OLLAMA_HOST_COLAB.get_secret_value(),
        supabase_url=settings.SUPABASE_URL.get_secret_value(),
        supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY.get_secret_value(),
        embedding_model=settings.EMBEDDING_MODEL,
        tavily_api_key=settings.TAVILY_API_KEY.get_secret_value(),
    )


def _create_evaluator() -> Evaluator:
    """Create an Evaluator instance from application settings."""
    return Evaluator(
        model=Model.GPT_OSS_120,
        provider=Provider.OLLAMA_CLOUD,
        api_key=settings.OLLAMA_API_KEY,
    )


def extract_code_from_response(response: str) -> str:
    """
    Extract Python code from Crafter's markdown response.

    Args:
        response: Raw LLM response containing fenced code blocks

    Returns:
        Extracted Python code string

    Raises:
        ValueError: If no code block found in the response
    """
    # Try ```python first, then bare ```
    for pattern in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:  
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    raise ValueError("No code block found in Crafter response")


def inject_input(code: str, new_input: str) -> str:
    """
    Replace the test input in generated agent code with a new input string.

    Handles two patterns commonly produced by the Crafter:
    1. Variable assignment: query = "original text"
    2. Inline content: "content": "original text"

    Args:
        code: Generated Python code from Crafter
        new_input: New input string to inject

    Returns:
        Modified code with the test input replaced

    Raises:
        ValueError: If no replaceable input pattern found
    """
    escaped = new_input.replace("\\", "\\\\").replace('"', '\\"')

    # Pattern 1: variable assignment (query = "...", question = "...", etc.)
    pattern_var = (
        r'((?:query|question|user_input|prompt)\s*=\s*)'
        r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\')'
    )
    if re.search(pattern_var, code):
        return re.sub(pattern_var, rf'\1"{escaped}"', code, count=1)

    # Pattern 2: inline "content": "..."
    pattern_inline = (
        r'("content"\s*:\s*)'
        r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"]*"|\'[^\']*\')'
    )
    if re.search(pattern_inline, code):
        return re.sub(pattern_inline, rf'\1"{escaped}"', code, count=1)

    raise ValueError(
        "Could not find input pattern to replace in generated code. "
        "Expected 'query = \"...\"' or '\"content\": \"...\"' pattern."
    )


async def run_execution_cases(
    code: str, cases: list[dict]
) -> list[dict]:
    """
    Run execution cases concurrently in Docker sandboxes.

    Each case gets the generated code with its input injected,
    then runs in an isolated Docker container.

    Args:
        code: Base generated code (before input injection)
        cases: List of execution case dicts from dataset

    Returns:
        List of result dicts with case_id, stdout, stderr, etc.
    """
    executor = CodeExecutor()

    async def _run_single(case: dict) -> dict:
        case_id = case["case_id"]
        input_value = case["input"]["input"]

        try:
            modified_code = inject_input(code, input_value)
        except ValueError as e:
            logger.error("Input injection failed for %s: %s", case_id, e)
            return {
                "case_id": case_id,
                "input": input_value,
                "reference_answer": case.get("reference_answer", ""),
                "execution_result": {
                    "stdout": "",
                    "stderr": str(e),
                    "exit_code": -1,
                    "timed_out": False,
                    "execution_time": 0.0,
                },
            }

        result = await asyncio.to_thread(
            executor.execute, modified_code, SANDBOX_TIMEOUT
        )

        return {
            "case_id": case_id,
            "input": input_value,
            "reference_answer": case.get("reference_answer", ""),
            "execution_result": result,
        }

    tasks = [_run_single(case) for case in cases]
    return await asyncio.gather(*tasks, return_exceptions=True)


def evaluate_cases(
    evaluator: Evaluator,
    task_type: str,
    execution_results: list[dict],
) -> list[dict]:
    """
    Evaluate each execution result with the LLM-as-judge.

    Args:
        evaluator: Configured Evaluator instance
        task_type: Task type string for evaluation prompt
        execution_results: Results from run_execution_cases

    Returns:
        List of case result dicts with normalized scores (0-1)
    """
    case_results = []

    for result in execution_results:
        # asyncio.gather with return_exceptions=True can return exceptions
        if isinstance(result, Exception):
            logger.error("Execution case raised exception: %s", result)
            case_results.append({
                "case_id": "unknown",
                "status": "execution_error",
                "error": str(result),
                "score": 0.0,
            })
            continue

        exec_result = result["execution_result"]
        case_id = result["case_id"]

        if exec_result["exit_code"] != 0 or exec_result["timed_out"]:
            logger.warning(
                "Case %s failed: exit_code=%s timed_out=%s",
                case_id, exec_result["exit_code"], exec_result["timed_out"],
            )
            case_results.append({
                "case_id": case_id,
                "input": result["input"],
                "status": "execution_failed",
                "exit_code": exec_result["exit_code"],
                "stderr": exec_result["stderr"][:500],
                "timed_out": exec_result["timed_out"],
                "execution_time": exec_result["execution_time"],
                "score": 0.0,
            })
            continue

        # Successful execution — evaluate with LLM judge
        agent_output = exec_result["stdout"].strip()

        try:
            raw_score = evaluator.evaluate(
                task_type=task_type,
                query=result["input"],
                response=agent_output,
                reference_answer=result["reference_answer"],
            )
            normalized_score = round(raw_score / MAX_SCORE, 3)
        except Exception as e:
            logger.error("Evaluation failed for case %s: %s", case_id, e)
            case_results.append({
                "case_id": case_id,
                "input": result["input"],
                "status": "evaluation_error",
                "error": str(e),
                "agent_output": agent_output[:2000],
                "score": 0.0,
            })
            continue

        case_results.append({
            "case_id": case_id,
            "input": result["input"],
            "status": "evaluated",
            "agent_output": agent_output[:2000],
            "reference_answer": result["reference_answer"],
            "score": normalized_score,
            "execution_time": exec_result["execution_time"],
        })

    return case_results


def write_results(task_id: str, results: dict) -> str:
    """
    Write evaluation results to a JSON file.

    Args:
        task_id: Task identifier used in filename
        results: Complete results dict

    Returns:
        Absolute path to the written results file
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_id}_{timestamp}.json"
    filepath = os.path.join(RESULTS_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return filepath


async def run_pipeline(task: dict) -> dict:
    """
    Run the full evaluation pipeline for a single dataset task.

    Phases:
        1. Generate agent code via Crafter.ask()
        2. Inject inputs and execute concurrently in Docker sandbox
        3. Evaluate outputs with LLM-as-judge
        4. Write results to JSON file

    Args:
        task: A single task dict from EVAL_DATASET

    Returns:
        Complete results dict with all case scores and metadata
    """
    task_id = task["task_id"]
    query = task["query"]
    task_type = task.get("task_type", "general")

    # Phase 1: Generate code
    logger.info("[Phase 1] Generating code for task '%s'", task_id)
    crafter = _create_crafter()
    raw_response = await asyncio.to_thread(crafter.ask, query)
    code = extract_code_from_response(raw_response)
    logger.info(
        "[Phase 1] Complete — extracted %d chars of code", len(code)
    )

    # Build execution cases list
    cases = task["execution_cases"]

    # Phase 2: Execute all cases concurrently in sandbox
    logger.info("[Phase 2] Running %d execution cases", len(cases))
    execution_results = await run_execution_cases(code, cases)
    logger.info("[Phase 2] Complete — all cases finished")

    # Phase 3: Evaluate outputs
    logger.info("[Phase 3] Evaluating results with LLM judge")
    evaluator = _create_evaluator()
    case_results = evaluate_cases(evaluator, task_type, execution_results)

    # Aggregate scores
    scores = [r["score"] for r in case_results]
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    pass_threshold = task.get("scoring", {}).get("pass_threshold", 0.8)

    results = {
        "task_id": task_id,
        "task_type": task_type,
        "query": query,
        "generated_code": code,
        "case_results": case_results,
        "aggregate_score": avg_score,
        "pass_threshold": pass_threshold,
        "passed": avg_score >= pass_threshold,
        "timestamp": datetime.now().isoformat(),
    }

    # Phase 4: Write results
    filepath = write_results(task_id, results)
    logger.info("[Phase 4] Results written to %s", filepath)
    logger.info(
        "Task '%s' — Score: %.3f — Passed: %s",
        task_id, avg_score, results["passed"],
    )

    return results


async def main():
    """Run evaluation pipeline for the wiki task."""
    wiki_task = next(
        (t for t in EVAL_DATASET if t["task_id"] == "agent_wiki_001"),
        None,
    )
    if wiki_task is None:
        logger.error("Task 'agent_wiki_001' not found in dataset")
        return

    results = await run_pipeline(wiki_task)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"Task: {results['task_id']}")
    print(f"Score: {results['aggregate_score']:.3f}")
    print(f"Passed: {results['passed']}")
    print(f"{'=' * 60}")
    for case in results["case_results"]:
        status = "PASS" if case["score"] >= results["pass_threshold"] else "FAIL"
        print(f"  [{status}] {case['case_id']}: {case['score']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())