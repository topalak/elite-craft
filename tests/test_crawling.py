"""
Unit tests for Crawling.
"""

from unittest.mock import Mock, patch
import pytest

from elite_craft.services.crawling import crawl

class TestExtractSource:
    """Test the _extract_source helper function."""

    #def test_known_domain_returns_source_name()
    #def test_unknown_domain_raises_value_error()
    #def test_different_langchain_domains()

    def test_known_domain_returns_source_name(self):
        from elite_craft.services.crawling import _extract_source

        """Test that known domains return correct source names."""
        urls = [
            "https://docs.langchain.com/guide",
            "https://python.langchain.com/docs",
        ]

        results = [_extract_source(url) for url in urls]

        assert results[0] == 'langchain'
        assert results[1] == 'langchain'

    def test_unknown_domain_raises_value_error(self):
        from elite_craft.services.crawling import _extract_source

        """Test that unknown domains raise ValueError."""
        with pytest.raises(ValueError, match="Domain 'docs.docling' is not in SOURCE_MAPPING"):
            _extract_source("https://docs.docling/guide")



#class TestCrawl:
    """Test the main crawl async function."""

class TestCrawl:
    """Test the main crawl async function."""

    async def test_crawl_known_source(self):
        """Test that known domains return correct source name."""

        # This replaces the REAL AsyncWebCrawler with a fake one
        with patch('elite_craft.services.crawling.AsyncWebCrawler') as MockCrawler:

            # STEP 2: Set up the fake crawler that will be returned by __aenter__
            # Think: What object do we get inside "async with ... as crawler:"?
            # The actual reason that we use __aenter__, crawl method uses async with
            mock_crawler_instance = MockCrawler.return_value.__aenter__.return_value

            # STEP 3: Setup what crawler.arun() should return
            # Create a fake response object with a markdown attribute
            mock_response = Mock()
            mock_response.markdown = "# Fake LangChain content"
            mock_crawler_instance.arun.return_value = mock_response

            # STEP 4: Actually call the REAL crawl function (which uses our mocked AsyncWebCrawler)
            result = await crawl("https://docs.langchain.com/guide")

            # STEP 5: Assert the result is correct
            assert result["source"] == "langchain"  # Known domain!
            assert result["body_text"] == "# Fake LangChain content"
            assert result["url"] == "https://docs.langchain.com/guide"
            assert "crawled_time" in result

    async def test_crawl_unknown_source(self):
        """Test crawl sets source to UNKNOWN for unrecognized domains."""
        with patch('elite_craft.services.crawling.AsyncWebCrawler') as MockCrawler:
            # Setup mock crawler behavior
            mock_crawler_instance = MockCrawler.return_value.__aenter__.return_value
            mock_response = Mock()
            mock_response.markdown = "# Fake content from unknown domain"
            mock_crawler_instance.arun.return_value = mock_response

            # Act: Call crawl with unknown domain
            result = await crawl("https://unknown-docs.example.com/guide")

            # Assert: source should be "UNKNOWN"
            assert result["source"] == "UNKNOWN"
            assert result["body_text"] == "# Fake content from unknown domain"
            assert result["url"] == "https://unknown-docs.example.com/guide"
            assert "crawled_time" in result
