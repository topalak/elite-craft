"""
Diagnostic test script to analyze retrieval quality.

Tests multiple queries and prints similarity scores to identify
retrieval issues and determine optimal thresholds.
"""
import logging
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import settings
from elite_craft.tools.retriever import Retriever

# Configure logging to see debug info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define test queries for different topics
TEST_QUERIES = [
    {
        "query": "explain user, ai, system messages",
        "expected_url_contains": "messages",
        "topic": "Messages"
    },
    {
        "query": "how do agents work in langgraph",
        "expected_url_contains": "agents",
        "topic": "Agents"
    },
    {
        "query": "what models are available in langchain",
        "expected_url_contains": "models",
        "topic": "Models"
    },
    {
        "query": "HumanMessage SystemMessage AIMessage",
        "expected_url_contains": "messages",
        "topic": "Message Classes"
    },
    {
        "query": "tool calling and function invocation",
        "expected_url_contains": "agents",
        "topic": "Tool Calling"
    }
]


def analyze_retrieval_quality():
    """Test retrieval quality across multiple queries."""

    print("=" * 80)
    print("RETRIEVAL QUALITY DIAGNOSTIC TEST")
    print("=" * 80)
    print()

    # Initialize retriever
    retriever = Retriever(
        supabase_url=settings.SUPABASE_URL,
        supabase_api_key=settings.SUPABASE_SERVICE_ROLE_SECRET_KEY,
        embedding_model_name=settings.EMBEDDING_MODEL
    )

    for test_case in TEST_QUERIES:
        query = test_case["query"]
        expected = test_case["expected_url_contains"]
        topic = test_case["topic"]

        print(f"\n{'─' * 80}")
        print(f"TEST: {topic}")
        print(f"{'─' * 80}")
        print(f"Query: '{query}'")
        print(f"Expected URL should contain: '{expected}'")
        print()

        # Retrieve with very low threshold to see all results
        chunks = retriever.retrieve_relevant_chunks(
            query=query,
            match_count=10,
            threshold=0.0  # Get everything to analyze
        )

        if not chunks:
            print("❌ No chunks retrieved!")
            continue

        print(f"Retrieved {len(chunks)} chunks:\n")

        # Analyze each chunk
        correct_count = 0
        for idx, chunk in enumerate(chunks, 1):
            url = chunk.get('url', 'N/A')
            similarity = chunk.get('similarity', 0)
            content_preview = chunk.get('content', '')[:80].replace('\n', ' ')

            # Check if URL matches expected
            is_correct = expected in url.lower()
            if is_correct:
                correct_count += 1
                marker = "✓"
                color = "\033[92m"  # Green
            else:
                marker = "✗"
                color = "\033[91m"  # Red

            reset = "\033[0m"

            print(f"{color}{marker}{reset} [{idx}] Similarity: {similarity:.4f}")
            print(f"    URL: {url}")
            print(f"    Content: {content_preview}...")
            print()

        # Summary for this query
        accuracy = (correct_count / len(chunks)) * 100
        print(f"Summary: {correct_count}/{len(chunks)} chunks from correct URL ({accuracy:.1f}% accuracy)")

        # Determine if threshold would help
        if chunks:
            max_wrong_similarity = max(
                [c['similarity'] for c in chunks if expected not in c.get('url', '').lower()],
                default=0
            )
            min_correct_similarity = min(
                [c['similarity'] for c in chunks if expected in c.get('url', '').lower()],
                default=1.0
            )

            print(f"Highest similarity for WRONG URL: {max_wrong_similarity:.4f}")
            print(f"Lowest similarity for CORRECT URL: {min_correct_similarity:.4f}")

            if min_correct_similarity > max_wrong_similarity:
                suggested_threshold = (min_correct_similarity + max_wrong_similarity) / 2
                print(f"✓ Threshold could help! Suggested: {suggested_threshold:.4f}")
            else:
                print("✗ Threshold alone won't solve this - embeddings aren't discriminating well")

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    analyze_retrieval_quality()