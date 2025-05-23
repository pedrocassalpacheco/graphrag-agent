import os
import pytest
from dotenv import load_dotenv

from graphrag_agent.tools.document_embedding import (
    OllamaEmbeddingProcessor,
    OpenAIEmbeddingProcessor,
)

# Load .env.test for OpenAI key
load_dotenv(dotenv_path=".env.test")


@pytest.mark.asyncio
async def test_ollama_embedding():
    processor = OllamaEmbeddingProcessor(
        model_name="nomic-embed-text", api_host="http://localhost:11434"
    )
    doc = {"title": "Ollama", "content": "This is a test for Ollama embedding."}
    result = await processor._process_item(doc.copy())
    assert "embedding" in result
    assert isinstance(result["embedding"], list)
    assert all(isinstance(x, float) for x in result["embedding"])
    assert len(result["embedding"]) > 0


@pytest.mark.asyncio
async def test_openai_embedding():
    api_key = os.getenv("OPENAI_API_KEY")
    assert api_key, "OPENAI_API_KEY must be set in .env.test"
    processor = OpenAIEmbeddingProcessor(
        model_name="text-embedding-3-large", api_key=api_key
    )
    doc = {"title": "OpenAI", "content": "This is a test for OpenAI embedding."}
    result = await processor._process_item(doc.copy())
    assert "$vector" in result
    assert isinstance(result["$vector"], list)
    assert all(isinstance(x, float) for x in result["$vector"])
    assert len(result["$vector"]) > 0
