import asyncio
import json
import io
import os
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, replace
from typing import Dict, List, Union, Any, Optional, override

from graphrag_agent.tools.async_processor import BaseAsyncProcessor
from graphrag_agent.tools.parse_content import ParsedContent
from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class AsyncDocumentEmbeddingProcessor(BaseAsyncProcessor, ABC):
    """
    Base class for async document embedding processors.
    Follows the same pattern as crawlers and parsers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _initialize_model(self) -> None:
        """Initialize the embedding model. Override in subclasses."""
        pass

    def _prepare_text(self, document: ParsedContent) -> str:
        """Prepare text for embedding by concatenating source, section, and content."""
        source = document.source or ""
        section = document.section or ""
        content = document.content or ""

        return "\n".join([source, section, content])

    @abstractmethod
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text. Must be implemented by subclasses."""
        pass

    @override
    async def _process_item(self, item: ParsedContent) -> ParsedContent:
        """Process a ParsedContent item by adding an embedding."""
        try:
            # Prepare text from the document
            text = self._prepare_text(item)

            # Generate embedding
            embedding = await self._generate_embedding(text)

            # Create new ParsedContent with embedding (immutable)
            result = replace(item, vector=embedding)

            self.processed_count += 1
            logger.debug(
                f"Generated embedding for document {self.processed_count}: {item.source} - {item.section}"
            )

            return result

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            logger.debug(f"Call stack:\n{traceback.format_exc()}")
            # Return original item on error
            return item

    async def _cleanup(self) -> None:
        """Override in subclasses if any resource needs cleanup."""
        logger.debug(f"Cleaning up {self.__class__.__name__}")


class OpenAIEmbeddingProcessor(AsyncDocumentEmbeddingProcessor):
    """OpenAI embedding processor using text-embedding models."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        dimensions: int = 3072,
        max_text_length: int = 8191,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.dimensions = dimensions
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client: Optional[Any] = None
        self.max_text_length = max_text_length

    async def _initialize_model(self) -> None:
        """Initialize OpenAI client."""
        if self.client is None:
            try:
                logger.info(f"Initializing OpenAI client for model: {self.model_name}")
                from openai import AsyncOpenAI

                if not self.api_key:
                    raise ValueError(
                        "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
                    )

                self.client = AsyncOpenAI(api_key=self.api_key)
                logger.debug("OpenAI client initialization complete")

            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
                logger.debug(f"Call stack:\n{traceback.format_exc()}")
                raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        if self.client is None:
            await self._initialize_model()

        if not text.strip():
            return [0.0] * self.dimensions

        try:
            # Truncate text if too long
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
            logger.debug(f"Call stack:\n{traceback.format_exc()}")
            return [0.0] * self.dimensions

    async def _cleanup(self) -> None:
        """Cleanup OpenAI client."""
        if self.client:
            await self.client.close()
            self.client = None
        await super()._cleanup()


class OllamaEmbeddingProcessor(AsyncDocumentEmbeddingProcessor):
    """Ollama embedding processor for local models."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        api_host: str = "http://localhost:11434",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.api_host = api_host
        self.client: Optional[Any] = None

    async def _initialize_model(self) -> None:
        """Initialize Ollama client."""
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
        """Generate embedding using Ollama API."""
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
            logger.debug(f"Call stack:\n{traceback.format_exc()}")
            return []

    async def _cleanup(self) -> None:
        """Cleanup Ollama client."""
        # Ollama client doesn't need explicit cleanup
        await super()._cleanup()


class HuggingFaceEmbeddingProcessor(AsyncDocumentEmbeddingProcessor):
    """HuggingFace embedding processor using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.model: Optional[Any] = None

    async def _initialize_model(self) -> None:
        """Initialize HuggingFace sentence-transformers model."""
        if self.model is None:
            try:
                logger.info(f"Initializing HuggingFace model: {self.model_name}")
                from sentence_transformers import SentenceTransformer

                # Run model initialization in thread to avoid blocking
                self.model = await asyncio.to_thread(
                    SentenceTransformer, self.model_name, device=self.device
                )
                logger.info("HuggingFace model initialization complete")

            except Exception as e:
                logger.error(f"Error initializing HuggingFace model: {str(e)}")
                logger.error(f"Call stack:\n{traceback.format_exc()}")
                raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using HuggingFace sentence-transformers."""
        if self.model is None:
            await self._initialize_model()

        if not text.strip():
            # Return zero vector with model dimensions
            return [0.0] * self.model.get_sentence_embedding_dimension()

        try:
            # Run embedding generation in thread to avoid blocking
            embedding = await asyncio.to_thread(
                self.model.encode, text, normalize_embeddings=self.normalize_embeddings
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating embedding with HuggingFace: {str(e)}")
            logger.debug(f"Call stack:\n{traceback.format_exc()}")
            return [0.0] * self.model.get_sentence_embedding_dimension()

    async def _cleanup(self) -> None:
        """Cleanup HuggingFace model."""
        if self.model:
            # Move model cleanup to thread if needed
            try:
                await asyncio.to_thread(delattr, self, "model")
            except:
                pass
            self.model = None
        await super()._cleanup()
