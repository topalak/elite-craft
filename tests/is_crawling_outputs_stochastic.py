import asyncio
import difflib

from elite_craft.services.crawling import crawl


def analyze_text_differences(reference_text: str, current_text: str, url: str, crawl_index: int) -> dict:
    """
    Analyze differences between two texts with multiple normalization strategies.

    Args:
        reference_text: The baseline text
        current_text: The new text to compare
        url: URL being compared
        crawl_index: Crawl iteration number

    Returns:
        Dictionary with analysis results and similarity metrics
    """
    # 1. Basic statistics
    stats = {
        "url": url,
        "crawl_index": crawl_index,
        "reference_length": len(reference_text),
        "current_length": len(current_text),
        "length_diff": len(current_text) - len(reference_text),
    }

    # 2. Raw comparison
    stats["exactly_equal"] = reference_text == current_text
    stats["raw_similarity"] = difflib.SequenceMatcher(None, reference_text, current_text).ratio()

    # 3. Whitespace normalization strategies
    def normalize_spaces(text: str) -> str:
        """Replace all whitespace with single space"""
        return ' '.join(text.split())

    def remove_all_whitespace(text: str) -> str:
        """Remove all whitespace characters"""
        return ''.join(text.split())

    def normalize_lines(text: str) -> str:
        """Normalize line endings and remove empty lines"""
        return '\n'.join(line.strip() for line in text.splitlines() if line.strip())

    # Apply normalizations
    ref_normalized_spaces = normalize_spaces(reference_text)
    cur_normalized_spaces = normalize_spaces(current_text)
    stats["equal_after_space_normalization"] = ref_normalized_spaces == cur_normalized_spaces
    stats["similarity_after_space_norm"] = difflib.SequenceMatcher(None, ref_normalized_spaces, cur_normalized_spaces).ratio()

    ref_no_whitespace = remove_all_whitespace(reference_text)
    cur_no_whitespace = remove_all_whitespace(current_text)
    stats["equal_without_whitespace"] = ref_no_whitespace == cur_no_whitespace
    stats["similarity_without_whitespace"] = difflib.SequenceMatcher(None, ref_no_whitespace, cur_no_whitespace).ratio()

    ref_normalized_lines = normalize_lines(reference_text)
    cur_normalized_lines = normalize_lines(current_text)
    stats["equal_after_line_normalization"] = ref_normalized_lines == cur_normalized_lines

    # 4. Character-level analysis
    ref_chars = set(reference_text)
    cur_chars = set(current_text)
    stats["unique_chars_reference"] = len(ref_chars)
    stats["unique_chars_current"] = len(cur_chars)
    stats["new_chars_in_current"] = list(cur_chars - ref_chars)
    stats["missing_chars_from_reference"] = list(ref_chars - cur_chars)

    # 5. Line-by-line comparison
    ref_lines = reference_text.splitlines()
    cur_lines = current_text.splitlines()
    stats["reference_line_count"] = len(ref_lines)
    stats["current_line_count"] = len(cur_lines)
    stats["line_count_diff"] = len(cur_lines) - len(ref_lines)

    # 6. Find what's different using SequenceMatcher
    # This shows actual character-level differences
    matcher = difflib.SequenceMatcher(None, reference_text, current_text)
    differences = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            differences.append({
                'type': 'changed',
                'reference': reference_text[i1:i2],
                'current': current_text[j1:j2]
            })
        elif tag == 'delete':
            differences.append({
                'type': 'removed',
                'content': reference_text[i1:i2]
            })
        elif tag == 'insert':
            differences.append({
                'type': 'added',
                'content': current_text[j1:j2]
            })

    stats["total_differences"] = len(differences)
    stats["differences_sample"] = differences[:10]  # First 10 differences

    return stats

def does_sites_contain_ai_generated(urls:list[str], max_attempts:int):

    total_ai_generated = 0
    dict_of_urls = {url: 0 for url in urls}
    for i in range(max_attempts):
        print(f"{i}. TIME FOR OUTER LOOP")

        contains = 0
        for url in urls:
            crawled_website = asyncio.run(crawl(url))

            raw_content = crawled_website['body_text']
            if "Responses are generated using AI and may contain mistakes" in raw_content:
                dict_of_urls[url] += 1
                contains += 1

        print(f"{i} OUTER HAS DONE WITH {contains} (Ai generated) / {len(urls)} URLS")
        total_ai_generated += contains

    print(dict_of_urls)
    print(f"TOTAL OUTER HAS DONE WITH {total_ai_generated} in {len(urls)*max_attempts} URLS")

async def are_sites_same(urls: list[str], upper_limit_for_loop: int, max_tries:int) -> bool:
    total_ai_generated = 0
    total_mismatch_counter = 0
    dict_of_urls = {url: 0 for url in urls}
    for url in urls:
        print(f"--- Analyzing stability: {url} ---")

        reference_text = None
        retries = 0
        while retries < max_tries:

            # 1. Get the Reference (Baseline) Content
            print(f"Fetching reference crawl (1/{upper_limit_for_loop})...")
            reference_result = await crawl(url)
            reference_text = reference_result['body_text']

            if "Responses are generated using AI and may contain mistakes" in reference_text:
                print(f"reference site '{url}' has 'Responses are generated using AI and may contain mistakes' sentences in it.")
                print(f"It will try to crawl reference site {url} again.")
                dict_of_urls[url] += 1
                total_ai_generated += 1
                retries += 1
                continue
            else:
                print(f"Reference site '{url}' isn't containing 'Responses are generated using AI and may contain mistakes' message, which we want.")
                break

        # Check if we exhausted all retries without getting valid reference
        if retries == max_tries and "Responses are generated using AI and may contain mistakes" in reference_text:
            print(f"'{url}' still contains 'Responses are generated using AI and may contain mistakes' message after {max_tries} attempts. Skipping this URL.")
            continue

        # 2. Check subsequent crawls against the reference
        # We range(upper_limit_for_loop - 1) because we already did 1 crawl
        mismatch_counter = 0
        for i in range(upper_limit_for_loop - 1):
            print(f"Comparing crawl {i + 2}/{upper_limit_for_loop}...")

            current_result = await crawl(url)
            current_text = current_result['body_text']


            # 3. Immediate Comparison
            if reference_text != current_text:
                print(f"⚠️  MISMATCH detected on {url}, analyzing differences...")

                # Analyze the differences
                stats = analyze_text_differences(reference_text, current_text, url, i + 2)

                # Print detailed statistics report
                print(f"\n{'='*60}")
                print(f"Difference Analysis Report")
                print(f"{'='*60}")
                print(f"URL: {url}")
                print(f"Crawl Index: {i + 2}")
                print(f"\nBasic Statistics:")
                print(f"  Reference length: {stats['reference_length']:,} chars")
                print(f"  Current length:   {stats['current_length']:,} chars")
                print(f"  Difference:       {stats['length_diff']:+,} chars")
                print(f"\nSimilarity Metrics:")
                print(f"  Exactly equal:                    {stats['exactly_equal']}")
                print(f"  Raw similarity:                   {stats['raw_similarity']*100:.2f}%")
                print(f"  Equal after space normalization:  {stats['equal_after_space_normalization']}")
                print(f"  Similarity after space norm:      {stats['similarity_after_space_norm']*100:.2f}%")
                print(f"  Equal without any whitespace:     {stats['equal_without_whitespace']}")
                print(f"  Similarity without whitespace:    {stats['similarity_without_whitespace']*100:.2f}%")
                print(f"  Equal after line normalization:   {stats['equal_after_line_normalization']}")
                print(f"\nLine Analysis:")
                print(f"  Reference lines: {stats['reference_line_count']}")
                print(f"  Current lines:   {stats['current_line_count']}")
                print(f"  Line count diff: {stats['line_count_diff']:+,}")

                print(f"\nDifferences Found:")
                print(f"  Total differences: {stats['total_differences']}")

                if stats['differences_sample']:
                    print(f"\nFirst {min(10, len(stats['differences_sample']))} Differences:")
                    for idx, diff in enumerate(stats['differences_sample'], 1):
                        if diff['type'] == 'added':
                            content_preview = diff['content'][:150].replace('\n', '\\n')
                            print(f"\n  [{idx}] ADDED:")
                            print(f"      {content_preview}")
                        elif diff['type'] == 'removed':
                            content_preview = diff['content'][:150].replace('\n', '\\n')
                            print(f"\n  [{idx}] REMOVED:")
                            print(f"      {content_preview}")
                        elif diff['type'] == 'changed':
                            ref_preview = diff['reference'][:100].replace('\n', '\\n')
                            cur_preview = diff['current'][:100].replace('\n', '\\n')
                            print(f"\n  [{idx}] CHANGED:")
                            print(f"      FROM: {ref_preview}")
                            print(f"      TO:   {cur_preview}")

                print(f"\n{'='*60}\n")

                mismatch_counter += 1
                total_mismatch_counter += 1

                if "Responses are generated using AI and may contain mistakes" in current_text:
                    print('   ⚠️  Current site has "Responses are generated using AI and may contain mistakes" message.')
                    total_ai_generated += 1
            else:
                print(f"✅ Crawl {i + 2}/{upper_limit_for_loop}: '{url}' IDENTICAL to reference")

        print(f"{url}'s results has arrived .You can analyze from here: {dict_of_urls}")
    print("✅ All sites remained identical across all checks.")
    print(f"Total number of 'Ai generated' URLs: {total_ai_generated}'")
    print(f"Total number of mismatches {total_mismatch_counter} except 'Ai generated response'")
    print(f"Final situation for 'Ai generated response' {dict_of_urls}")
    return True

if __name__ == '__main__':

    langchain_urls = [
        "https://docs.langchain.com/oss/python/langchain/agents",
        "https://docs.langchain.com/oss/python/langchain/messages",
        "https://docs.langchain.com/oss/python/langchain/models",
        "https://docs.langchain.com/oss/python/langchain/tools",
        "https://docs.langchain.com/oss/python/langchain/structured-output",
        "https://docs.langchain.com/oss/python/langchain/middleware/built-in",
        "https://docs.langchain.com/oss/python/langchain/overview",
        "https://docs.langchain.com/oss/python/langchain/streaming",
        "https://docs.langchain.com/oss/python/langchain/guardrails",
        "https://docs.langchain.com/oss/python/langchain/runtime",
        #"https://docs.langchain.com/oss/python/langchain/context-engineering",
        #"https://docs.langchain.com/oss/python/langchain/human-in-the-loop",

    ]

    pydantic_urls = [
        "https://ai.pydantic.dev/",
        "https://ai.pydantic.dev/install/",
        "https://ai.pydantic.dev/multi-agent-applications/",
    ]

    response = asyncio.run(are_sites_same(urls=pydantic_urls, upper_limit_for_loop=5, max_tries=5))
    print(response)