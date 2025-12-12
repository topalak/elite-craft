"""
Golden dataset evaluation pipeline for chunk retrieval quality.

This module provides a comprehensive framework for evaluating retrieval quality
using LLM-generated queries and LLM-as-judge evaluation with structured rubrics.

Workflow:
1. Generate diverse queries using LLM (with manual review)
2. Execute retrieval for each query
3. Evaluate chunk quality using LLM-as-judge with rubric
4. Aggregate results and generate reports
"""
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from config import settings
from elite_craft.model_provider import ModelConfig
from elite_craft.tools.retriever import Retriever


logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChunkEvalRubric(BaseModel):
    """LLM follows this to evaluate retrieval quality."""

    # Step 1: Relevance filtering
    relevant_chunk_count: int = Field(
        description="How many chunks are actually relevant?")
    relevance_threshold_met: bool = Field(
        description="Are >= 60% of chunks relevant?")

    # Step 2: Coverage (only for relevant chunks)
    query_fully_covered: bool = Field(
        description="Do relevant chunks have ALL info needed to answer?")
   # missing_aspects: List[str] = Field(
    #    description="What key aspects are missing, if any?")

    # Step 3: Quality checks
    contains_contradictions: bool = Field(
        description="Do chunks contradict each other?")
    #has_outdated_info: bool = Field(
    #    description="Any deprecated/outdated patterns?")

    # Step 4: Ranking quality
    top_chunk_is_relevant: bool = Field(
        description="Is the #1 ranked chunk actually relevant?")

    # Overall
    passed: bool = Field(
        description="Overall: good enough to generate accurate answer?")
    failure_reason: Optional[str] = Field(
        default=None,
        description="If failed, why?"
    )


class EvalQuery(BaseModel):
    """Single evaluation query with metadata."""

    id: str = Field(description="Unique identifier for this query")
    query: str = Field(description="The actual query string")
    category: str = Field(
        description="Query category (e.g., 'how-to', 'conceptual', 'troubleshooting')"
    )
    expected_topics: List[str] = Field(
        description="Topics that should appear in results"
    )
    source_filter: Optional[str] = Field(
        default=None,
        description="Optional source filter (e.g., 'langchain', 'langgraph')"
    )


class RetrievalResult(BaseModel):
    """Results from retrieval execution."""

    query_id: str
    query: str
    retrieved_chunks: List[dict]
    chunk_count: int
    execution_time_ms: float


class EvaluationResult(BaseModel):
    """Complete evaluation result for one query."""

    query_id: str
    query: str
    category: str
    retrieval_result: RetrievalResult
    rubric_scores: ChunkEvalRubric
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across all queries."""

    total_queries: int
    passed_queries: int
    failed_queries: int
    pass_rate: float

    avg_relevant_chunk_count: float
    avg_chunk_count: float
    relevance_threshold_met_rate: float

    query_fully_covered_rate: float
    top_chunk_relevant_rate: float

    contradiction_rate: float
    outdated_info_rate: float

    avg_execution_time_ms: float

    failure_reasons: dict[str, int] = Field(
        description="Counts of each failure reason"
    )


# ============================================================================
# QUERY GENERATION
# ============================================================================

class QueryGenerator:
    """Generate diverse evaluation queries using LLM."""

    def __init__(self, llm_model_name: str = "gpt-oss:20b-cloud"):
        """
        Initialize query generator with LLM.

        Args:
            llm_model_name: Model to use for query generation
        """
        model_config = ModelConfig(
            model=llm_model_name,
            temperature=0.7,  # Higher temperature for diversity
            use_ollama_local=settings.USE_OLLAMA_LOCAL,
            api_key=settings.OLLAMA_API_KEY
        )
        self.llm = model_config.get_llm()

    def generate_queries(self, num_queries: int = 30) -> List[EvalQuery]:
        """
        Generate diverse evaluation queries.

        Args:
            num_queries: Number of queries to generate

        Returns:
            List of generated queries with metadata
        """
        prompt = f"""Generate {num_queries} diverse evaluation queries for testing a RAG system that helps developers build AI agents.

The knowledge base contains documentation about:
- LangChain (agent orchestration framework)
- LangGraph (graph-based agent workflows)

Generate queries across these categories:
1. How-to queries (e.g., "How do I add memory to my agent?")
2. Conceptual queries (e.g., "What's the difference between StateGraph and MessageGraph?")
3. Troubleshooting queries (e.g., "Why is my agent not streaming responses?")
4. Best practices queries (e.g., "What's the recommended way to handle errors in agents?")
5. Comparison queries (e.g., "When should I use LangGraph vs plain LangChain?")

Return a JSON array with this structure:
[
  {{
    "id": "query_001",
    "query": "How do I implement streaming in LangGraph?",
    "category": "how-to",
    "expected_topics": ["streaming", "LangGraph", "async"],
    "source_filter": "langgraph"
  }}
]

Make queries realistic and diverse. Vary complexity from simple to advanced.
ONLY return the JSON array, nothing else."""

        response = self.llm.invoke(prompt)

        # Parse JSON from response
        try:
            queries_data = json.loads(response.content)
            return [EvalQuery(**q) for q in queries_data]
        except Exception as e:
            logger.error(f"Failed to parse generated queries: {e}")
            raise


# ============================================================================
# RETRIEVAL EXECUTOR
# ============================================================================

class RetrievalExecutor:
    """Execute retrieval for evaluation queries."""

    def __init__(
        self,
        supabase_url: str,
        supabase_api_key: str,
        embedding_model_name: str,
        match_count: int = 8
    ):
        """
        Initialize retrieval executor.

        Args:
            supabase_url: Supabase project URL
            supabase_api_key: Supabase API key
            embedding_model_name: Embedding model name
            match_count: Number of chunks to retrieve
        """
        self.retriever = Retriever(
            supabase_url=supabase_url,
            supabase_api_key=supabase_api_key,
            embedding_model_name=embedding_model_name
        )
        self.match_count = match_count

    def execute(self, query: EvalQuery) -> RetrievalResult:
        """
        Execute retrieval for a single query.

        Args:
            query: Evaluation query to execute

        Returns:
            Retrieval results with timing info
        """
        import time
        start_time = time.time()

        chunks = self.retriever.retrieve_relevant_chunks(
            query=query.query,
            source_filter=query.source_filter
        )

        execution_time_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query_id=query.id,
            query=query.query,
            retrieved_chunks=chunks[:self.match_count],
            chunk_count=len(chunks[:self.match_count]),
            execution_time_ms=execution_time_ms
        )


# ============================================================================
# LLM-AS-JUDGE EVALUATOR
# ============================================================================

class ChunkQualityJudge:
    """LLM-as-judge for evaluating chunk quality using rubric."""

    def __init__(self, llm_model_name: str = "gpt-oss:20b-cloud"):
        """
        Initialize judge with LLM.

        Args:
            llm_model_name: Model to use for evaluation
        """
        model_config = ModelConfig(
            model=llm_model_name,
            temperature=0,  # Deterministic for evaluation
            use_ollama_local=settings.USE_OLLAMA_LOCAL,
            api_key=settings.OLLAMA_API_KEY
        )
        self.llm = model_config.get_llm().with_structured_output(ChunkEvalRubric)

    def evaluate(
        self,
        query: str,
        chunks: List[dict]
    ) -> ChunkEvalRubric:
        """
        Evaluate chunk quality for a query.

        Args:
            query: The search query
            chunks: Retrieved chunks to evaluate

        Returns:
            Rubric evaluation results
        """
        # Format chunks for prompt
        chunks_text = "\n\n".join([
            f"**Chunk {i+1}** (similarity: {chunk.get('similarity', 'N/A')})\n"
            f"Source: {chunk.get('source', 'unknown')}\n"
            f"Content: {chunk.get('content', '')[:500]}..."
            for i, chunk in enumerate(chunks)
        ])

        prompt = f"""Evaluate if the retrieved chunks can answer this query.
Return JSON matching the ChunkEvalRubric schema.

Query: {query}

Retrieved chunks ({len(chunks)} total):
{chunks_text}

Evaluation criteria:
1. Count how many chunks are actually relevant to the query
2. Check if >= 60% of chunks are relevant
3. Check if relevant chunks have ALL information needed to fully answer the query
4. List any missing key aspects (empty list if none)
5. Check for contradictions between chunks
6. Check for outdated/deprecated information
7. Check if the #1 ranked chunk is relevant
8. Overall: Can these chunks generate an accurate answer?
9. If failed, explain why in one sentence

Be strict: if chunks are missing key info, mark query_fully_covered=False.
Return ONLY valid JSON matching ChunkEvalRubric schema."""

        try:
            result = self.llm.invoke(prompt)
            return result
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            # Return a failed evaluation
            return ChunkEvalRubric(
                relevant_chunk_count=0,
                relevance_threshold_met=False,
                query_fully_covered=False,
                missing_aspects=["Evaluation error"],
                contains_contradictions=False,
                has_outdated_info=False,
                top_chunk_is_relevant=False,
                passed=False,
                failure_reason=f"Evaluation error: {str(e)}"
            )


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

class EvaluationPipeline:
    """Complete evaluation pipeline orchestrator."""

    def __init__(
        self,
        output_dir: str = "tests/evaluate_with_golden_dataset/results",
        judge_model: str = "gpt-oss:20b-cloud",
        generator_model: str = "gpt-oss:20b-cloud"
    ):
        """
        Initialize evaluation pipeline.

        Args:
            output_dir: Directory to save results
            judge_model: Model to use for evaluation
            generator_model: Model to use for query generation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.query_generator = QueryGenerator(llm_model_name=generator_model)
        self.retrieval_executor = RetrievalExecutor(
            supabase_url=settings.SUPABASE_URL,
            supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY,
            embedding_model_name=settings.EMBEDDING_MODEL
        )
        self.judge = ChunkQualityJudge(llm_model_name=judge_model)

    def generate_and_save_queries(
        self,
        num_queries: int = 30,
        filename: str = "generated_queries.json"
    ) -> List[EvalQuery]:
        """
        Generate queries and save to file for manual review.

        Args:
            num_queries: Number of queries to generate
            filename: Output filename

        Returns:
            List of generated queries
        """
        logger.info(f"Generating {num_queries} evaluation queries...")
        queries = self.query_generator.generate_queries(num_queries)

        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(
                [q.model_dump() for q in queries],
                f,
                indent=2
            )

        logger.info(f"Saved queries to {output_file}")
        logger.info("⚠️  IMPORTANT: Review and edit queries manually before running evaluation!")

        return queries

    def load_queries(self, filename: str = "generated_queries.json") -> List[EvalQuery]:
        """
        Load queries from file.

        Args:
            filename: Input filename

        Returns:
            List of evaluation queries
        """
        input_file = self.output_dir / filename
        with open(input_file, 'r') as f:
            queries_data = json.load(f)

        return [EvalQuery(**q) for q in queries_data]

    def run_evaluation(
        self,
        queries: List[EvalQuery],
        output_filename: str = None
    ) -> List[EvaluationResult]:
        """
        Run complete evaluation pipeline.

        Args:
            queries: Queries to evaluate
            output_filename: Optional custom output filename

        Returns:
            List of evaluation results
        """
        results = []

        logger.info(f"Running evaluation on {len(queries)} queries...")

        for i, query in enumerate(queries, 1):
            logger.info(f"[{i}/{len(queries)}] Evaluating: {query.query}")

            # Execute retrieval
            retrieval_result = self.retrieval_executor.execute(query)

            # Evaluate with judge
            rubric_scores = self.judge.evaluate(
                query=query.query,
                chunks=retrieval_result.retrieved_chunks
            )

            # Create result
            eval_result = EvaluationResult(
                query_id=query.id,
                query=query.query,
                category=query.category,
                retrieval_result=retrieval_result,
                rubric_scores=rubric_scores
            )

            results.append(eval_result)

            logger.info(
                f"  ✓ Passed: {rubric_scores.passed}, "
                f"Relevant: {rubric_scores.relevant_chunk_count}/{retrieval_result.chunk_count}"
            )

        # Save results
        if output_filename is None:
            output_filename = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_file = self.output_dir / output_filename
        with open(output_file, 'w') as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                indent=2
            )

        logger.info(f"Saved evaluation results to {output_file}")

        return results

    def aggregate_metrics(
        self,
        results: List[EvaluationResult]
    ) -> AggregatedMetrics:
        """
        Aggregate metrics from evaluation results.

        Args:
            results: Evaluation results to aggregate

        Returns:
            Aggregated metrics
        """
        total = len(results)
        passed = sum(1 for r in results if r.rubric_scores.passed)

        failure_reasons = {}
        for r in results:
            if not r.rubric_scores.passed and r.rubric_scores.failure_reason:
                reason = r.rubric_scores.failure_reason
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        return AggregatedMetrics(
            total_queries=total,
            passed_queries=passed,
            failed_queries=total - passed,
            pass_rate=passed / total if total > 0 else 0,

            avg_relevant_chunk_count=sum(
                r.rubric_scores.relevant_chunk_count for r in results
            ) / total if total > 0 else 0,
            avg_chunk_count=sum(
                r.retrieval_result.chunk_count for r in results
            ) / total if total > 0 else 0,
            relevance_threshold_met_rate=sum(
                1 for r in results if r.rubric_scores.relevance_threshold_met
            ) / total if total > 0 else 0,

            query_fully_covered_rate=sum(
                1 for r in results if r.rubric_scores.query_fully_covered
            ) / total if total > 0 else 0,
            top_chunk_relevant_rate=sum(
                1 for r in results if r.rubric_scores.top_chunk_is_relevant
            ) / total if total > 0 else 0,

            contradiction_rate=sum(
                1 for r in results if r.rubric_scores.contains_contradictions
            ) / total if total > 0 else 0,
            outdated_info_rate=sum(
                1 for r in results if r.rubric_scores.has_outdated_info
            ) / total if total > 0 else 0,

            avg_execution_time_ms=sum(
                r.retrieval_result.execution_time_ms for r in results
            ) / total if total > 0 else 0,

            failure_reasons=failure_reasons
        )

    def print_report(self, metrics: AggregatedMetrics):
        """
        Print human-readable evaluation report.

        Args:
            metrics: Aggregated metrics to print
        """
        print("\n" + "="*70)
        print("CHUNK RETRIEVAL EVALUATION REPORT")
        print("="*70)

        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"  Total Queries: {metrics.total_queries}")
        print(f"  Passed: {metrics.passed_queries}")
        print(f"  Failed: {metrics.failed_queries}")
        print(f"  Pass Rate: {metrics.pass_rate:.1%}")

        print(f"\n📝 RELEVANCE METRICS")
        print(f"  Avg Chunks Retrieved: {metrics.avg_chunk_count:.1f}")
        print(f"  Avg Relevant Chunks: {metrics.avg_relevant_chunk_count:.1f}")
        print(f"  Relevance Threshold Met (>=60%): {metrics.relevance_threshold_met_rate:.1%}")
        print(f"  Top Chunk Relevant: {metrics.top_chunk_relevant_rate:.1%}")

        print(f"\n✅ COVERAGE & QUALITY")
        print(f"  Query Fully Covered: {metrics.query_fully_covered_rate:.1%}")
        print(f"  Contains Contradictions: {metrics.contradiction_rate:.1%}")
        print(f"  Has Outdated Info: {metrics.outdated_info_rate:.1%}")

        print(f"\n⚡ PERFORMANCE")
        print(f"  Avg Execution Time: {metrics.avg_execution_time_ms:.0f}ms")

        if metrics.failure_reasons:
            print(f"\n❌ FAILURE REASONS")
            for reason, count in sorted(
                metrics.failure_reasons.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  {count:2d}x - {reason}")

        print("\n" + "="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pipeline = EvaluationPipeline()

    print("\n" + "="*70)
    print("CHUNK RETRIEVAL EVALUATION PIPELINE")
    print("="*70)
    print("\nChoose an option:")
    print("1. Generate new queries (LLM-assisted)")
    print("2. Run evaluation on existing queries")
    print("3. Full pipeline (generate + evaluate)")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        num_queries = int(input("Number of queries to generate (default 30): ") or "30")
        pipeline.generate_and_save_queries(num_queries=num_queries)
        print("\n✓ Queries generated. Please review and edit before running evaluation.")

    elif choice == "2":
        queries = pipeline.load_queries()
        print(f"\nLoaded {len(queries)} queries")

        confirm = input("Run evaluation? (y/n): ").strip().lower()
        if confirm == 'y':
            results = pipeline.run_evaluation(queries)
            metrics = pipeline.aggregate_metrics(results)
            pipeline.print_report(metrics)

    elif choice == "3":
        num_queries = int(input("Number of queries to generate (default 30): ") or "30")

        # Generate
        queries = pipeline.generate_and_save_queries(num_queries=num_queries)

        # Manual review
        print("\n⚠️  Please review generated_queries.json and edit as needed.")
        input("Press Enter when ready to continue with evaluation...")

        # Reload (in case edited)
        queries = pipeline.load_queries()

        # Evaluate
        results = pipeline.run_evaluation(queries)
        metrics = pipeline.aggregate_metrics(results)
        pipeline.print_report(metrics)

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
