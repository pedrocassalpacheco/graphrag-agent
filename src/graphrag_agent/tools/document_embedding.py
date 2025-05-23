import asyncio
import json
import io
from typing import Dict, List, Union, Any, Optional
import traceback
import numpy as np
import os
from pathlib import Path

from graphrag_agent.utils.logging_config import get_logger
from graphrag_agent.tools.async_processor import BaseAsyncProcessor

logger = get_logger(__name__)


class AsyncDocumentEmbeddingProcessor(BaseAsyncProcessor):
    """
    Async processor for generating embeddings for documents.
    Follows the same pattern as crawlers and parsers.
    """

    def __init__(self, embedding_key: str = "$vector", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_key = embedding_key

    async def _initialize_model(self):
        """To be implemented by subclasses if needed."""
        pass

    def _prepare_text(self, document: Dict[str, Any]) -> str:
        title = document.get("title", "")
        content = document.get("content", "")
        if isinstance(content, list):
            text_parts = [title]
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, list):
                    text_parts.append(" ".join(str(subitem) for subitem in item))
                elif isinstance(item, dict) and "image" in item:
                    alt = item["image"].get("alt", "")
                    if alt:
                        text_parts.append(alt)
                else:
                    text_parts.append(str(item))
            return " ".join(text_parts)
        elif isinstance(content, str):
            return f"{title} {content}"
        else:
            return f"{title} {str(content)}"

    async def _generate_embedding(self, text: str) -> List[float]:
        """To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

    async def _process_item(self, document: Dict[str, Any]) -> Dict[str, Any]:
        try:
            text = self._prepare_text(document)
            embedding = await self._generate_embedding(text)
            document[self.embedding_key] = embedding
            self.processed_count += 1
            logger.info(
                f"Generated embedding for document {self.processed_count}: {document.get('title', 'Untitled')}"
            )
            return document
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            return document

    async def _cleanup(self):
        """Override in subclasses if any resource needs cleanup."""
        pass


class SentenceTransformerEmbeddingProcessor(AsyncDocumentEmbeddingProcessor):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        embedding_key: str = "embedding",
        *args,
        **kwargs,
    ):
        super().__init__(embedding_key, *args, **kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    async def _initialize_model(self):
        if self.model is None:
            try:
                logger.info(
                    f"Initializing SentenceTransformer model: {self.model_name}"
                )
                from sentence_transformers import SentenceTransformer

                self.model = await asyncio.to_thread(
                    SentenceTransformer, self.model_name
                )
                logger.info("Model initialization complete")
            except Exception as e:
                logger.error(f"Error initializing SentenceTransformer model: {str(e)}")
                logger.error(f"Call stack:\n{traceback.format_exc()}")
                raise

    async def _generate_embedding(self, text: str) -> List[float]:
        if self.model is None:
            await self._initialize_model()
        if not text.strip():
            return [0.0] * self.model.get_sentence_embedding_dimension()
        try:
            embedding = await asyncio.to_thread(self.model.encode, text)
            return embedding.tolist()
        except Exception as e:
            logger.error(
                f"Error generating embedding with SentenceTransformer: {str(e)}"
            )
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            return [0.0] * self.model.get_sentence_embedding_dimension()


class OllamaEmbeddingProcessor(AsyncDocumentEmbeddingProcessor):
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        api_host: str = "http://localhost:11434",
        embedding_key: str = "embedding",
        *args,
        **kwargs,
    ):
        super().__init__(embedding_key, *args, **kwargs)
        self.model_name = model_name
        self.api_host = api_host
        self.client = None

    async def _initialize_model(self):
        if self.client is None:
            try:
                logger.info(f"Initializing Ollama client for model: {self.model_name}")
                from ollama import AsyncClient

                self.client = AsyncClient(host=self.api_host)
                logger.info("Ollama client initialization complete")
            except Exception as e:
                logger.error(f"Error initializing Ollama client: {str(e)}")
                logger.error(f"Call stack:\n{traceback.format_exc()}")
                raise

    async def _generate_embedding(self, text: str) -> List[float]:
        if self.client is None:
            await self._initialize_model()
        if not text.strip():
            return []
        try:
            response = await self.client.embeddings(model=self.model_name, prompt=text)
            if hasattr(response, "embedding"):
                return response.embedding
            elif isinstance(response, dict) and "embedding" in response:
                return response["embedding"]
            else:
                logger.error(f"Unexpected response format from Ollama: {response}")
                return []
        except Exception as e:
            logger.error(f"Error getting embedding from Ollama: {str(e)}")
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            return []


class OpenAIEmbeddingProcessor(AsyncDocumentEmbeddingProcessor):
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: str = None,
        dimensions: int = 3072,
        embedding_key: str = "$vector",
        max_text_length: int = 8191,
        *args,
        **kwargs,
    ):
        super().__init__(embedding_key, *args, **kwargs)
        self.model_name = model_name
        self.dimensions = dimensions
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        self.max_text_length = max_text_length

    async def _initialize_model(self):
        if self.client is None:
            try:
                logger.info(f"Initializing OpenAI client for model: {self.model_name}")
                from openai import AsyncOpenAI

                if not self.api_key:
                    raise ValueError(
                        "OpenAI API key not found. Set it in .env or pass it to the constructor."
                    )
                self.client = AsyncOpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialization complete")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
                logger.error(f"Call stack:\n{traceback.format_exc()}")
                raise

    async def _generate_embedding(self, text: str) -> List[float]:
        if self.client is None:
            await self._initialize_model()
        if not text.strip():
            return [0.0] * self.dimensions
        try:
            if len(text) > self.max_text_length * 4:
                logger.warning(
                    f"Text exceeds recommended length. Truncating from {len(text)} chars."
                )
                text = text[: self.max_text_length * 4]
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimensions,
            )
            if response and response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                logger.error(f"Empty or unexpected response from OpenAI: {response}")
                return [0.0] * self.dimensions
        except Exception as e:
            logger.error(f"Error getting embedding from OpenAI: {str(e)}")
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            return [0.0] * self.dimensions
