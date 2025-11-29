"""
Unit tests for ModelConfig.
"""

from unittest.mock import Mock, patch, MagicMock

from elite_craft.model_provider import ModelConfig


class TestModelConfigInit:
    """Test ModelConfig initialization."""

    def test_model_config_initializes_with_defaults(self):
        """Test that ModelConfig initializes with default values."""
        config = ModelConfig(model="test-model")

        assert config.model == "test-model"
        assert config.model_provider_url is None
        assert config.num_ctx is None
        assert config.reasoning is False
        assert config.temperature == 0
        assert config.use_ollama_local is False
        assert config.api_key is None
        assert config.use_groq is None

    def test_model_config_initializes_with_custom_values(self):
        """Test that ModelConfig initializes with custom values."""
        config = ModelConfig(
            model="custom-model",
            num_ctx=4096,
            model_provider_url="http://localhost:11434",
            reasoning=True,
            temperature=0.7,
            use_ollama_local=True,
            api_key="test-key",
            use_groq=False
        )

        assert config.model == "custom-model"
        assert config.num_ctx == 4096
        assert config.model_provider_url == "http://localhost:11434"
        assert config.reasoning is True
        assert config.temperature == 0.7
        assert config.use_ollama_local is True
        assert config.api_key == "test-key"
        assert config.use_groq is False


class TestGetLLM:
    """Test the get_llm method."""

    def test_get_llm_with_ollama_local(self):
        """Test get_llm returns ChatOllama when use_ollama_local=True."""

        with patch('elite_craft.model_provider._check_and_pull_ollama_model') as mock_check, \
             patch('elite_craft.model_provider.Client') as MockClient, \
             patch('elite_craft.model_provider.ChatOllama') as MockChatOllama:

            # Setup mocks
            mock_ollama_client = MockClient.return_value
            mock_ollama_client.generate.return_value = None
            mock_chat_ollama = MockChatOllama.return_value

            # Create config
            config = ModelConfig(
                model="llama3",
                model_provider_url="http://localhost:11434",
                num_ctx=4096,
                reasoning=True,
                temperature=0.5,
                use_ollama_local=True
            )

            # Call get_llm
            result = config.get_llm()

            # Verify _check_and_pull_ollama_model was called
            mock_check.assert_called_once_with(
                model_name="llama3",
                ollama_url="http://localhost:11434"
            )

            # Verify Ollama client was created
            MockClient.assert_called_once_with(host="http://localhost:11434")
            mock_ollama_client.generate.assert_called_once_with(model="llama3")

            # Verify ChatOllama was created with correct params
            MockChatOllama.assert_called_once_with(
                model="llama3",
                base_url="http://localhost:11434",
                num_ctx=4096,
                reasoning=True,
                temperature=0.5,
                keep_alive="5m"
            )

            assert result == mock_chat_ollama

    def test_get_llm_with_groq(self):
        """Test get_llm returns ChatGroq when use_groq=True."""

        with patch('elite_craft.model_provider.ChatGroq') as MockChatGroq:

            mock_chat_groq = MockChatGroq.return_value

            # Create config
            config = ModelConfig(
                model="llama-3.3-70b-versatile",
                api_key="groq-api-key",
                temperature=0.7,
                use_groq=True
            )

            # Call get_llm
            result = config.get_llm()

            # Verify ChatGroq was created with correct params
            MockChatGroq.assert_called_once_with(
                model="llama-3.3-70b-versatile",
                api_key="groq-api-key",
                temperature=0.7
            )

            assert result == mock_chat_groq

    def test_get_llm_with_ollama_cloud(self):
        """Test get_llm returns ChatOllama for cloud when neither local nor groq."""

        with patch('elite_craft.model_provider.ChatOllama') as MockChatOllama:

            mock_chat_ollama = MockChatOllama.return_value

            # Create config (neither use_ollama_local nor use_groq)
            config = ModelConfig(
                model="llama3",
                api_key="ollama-cloud-key",
                num_ctx=2048,
                reasoning=False,
                temperature=0.3,
                use_ollama_local=False,
                use_groq=False
            )

            # Call get_llm
            result = config.get_llm()

            # Verify ChatOllama was created with cloud config
            MockChatOllama.assert_called_once_with(
                model="llama3",
                base_url="https://ollama.com",
                client_kwargs={
                    'headers': {'Authorization': 'Bearer ollama-cloud-key'}
                },
                num_ctx=2048,
                reasoning=False,
                temperature=0.3,
                keep_alive="5m"
            )

            assert result == mock_chat_ollama


class TestGetEmbedding:
    """Test the get_embedding method."""

    def test_get_embedding_returns_ollama_embeddings(self):
        """Test get_embedding returns OllamaEmbeddings."""

        with patch('elite_craft.model_provider._check_and_pull_ollama_model') as mock_check, \
             patch('elite_craft.model_provider.Client') as MockClient, \
             patch('elite_craft.model_provider.OllamaEmbeddings') as MockOllamaEmbeddings:

            # Setup mocks
            mock_ollama_client = MockClient.return_value
            mock_ollama_client.embed.return_value = None
            mock_embeddings = MockOllamaEmbeddings.return_value

            # Create config
            config = ModelConfig(
                model="embeddinggemma",
                model_provider_url="http://localhost:11434"
            )

            # Call get_embedding
            result = config.get_embedding()

            # Verify _check_and_pull_ollama_model was called
            mock_check.assert_called_once_with(
                model_name="embeddinggemma",
                ollama_url="http://localhost:11434"
            )

            # Verify Ollama client was created
            MockClient.assert_called_once_with(host="http://localhost:11434")
            mock_ollama_client.embed.assert_called_once_with(model="embeddinggemma")

            # Verify OllamaEmbeddings was created
            MockOllamaEmbeddings.assert_called_once_with(
                model="embeddinggemma",
                base_url="http://localhost:11434"
            )

            assert result == mock_embeddings


class TestMainFunction:
    """Test the main function."""

    def test_main_prints_message(self, capsys):
        """Test that main() prints 'main'."""
        from elite_craft.model_provider import main

        main()

        captured = capsys.readouterr()
        assert "main" in captured.out