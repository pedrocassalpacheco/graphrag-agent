import asyncio
import json
import io
from typing import Dict, List, Union, Any, Optional
import traceback
import numpy as np

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseAsyncEmbeddingGenerator:
    """
    Base class for asynchronous embedding generators.
    
    This abstract class defines the interface for embedding generators that
    process documents from a queue and add embeddings to them.
    """
    
    def __init__(self, embedding_key: str = "$vector"):
        self.embedding_key = embedding_key
        self.processed_count = 0
    
    async def initialize_model(self):
        """Initialize the embedding model (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _prepare_text(self, document: Dict[str, Any]) -> str:
        """Extract and prepare text from a document for embedding."""
        title = document.get("title", "")
        content = document.get("content", "")
        
        # Handle different content types
        if isinstance(content, list):
            text_parts = [title]
            
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, list):
                    # Flatten nested lists
                    text_parts.append(" ".join(str(subitem) for subitem in item))
                elif isinstance(item, dict) and "image" in item:
                    # Skip images or include alt text if needed
                    alt = item["image"].get("alt", "")
                    if alt:
                        text_parts.append(alt)
                else:
                    # Convert any other type to string
                    text_parts.append(str(item))
            
            return " ".join(text_parts)
        elif isinstance(content, str):
            return f"{title} {content}"
        else:
            # Convert any non-string content to string
            return f"{title} {str(content)}"
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def embed(
        self,
        input: asyncio.Queue,
        output: Union[asyncio.Queue, Dict[str, Dict[str, Any]], io.TextIOWrapper] = None,
    ) -> None:
        """
        Process documents from input queue, generate embeddings, and output results.
        
        Args:
            input: Queue containing documents to process
            output: Destination for processed documents (queue, dict, or file)
        """
        # Initialize the model if not already done
        await self.initialize_model()
        
        while True:
            document = await input.get()
            
            if document is None:  # Signal that processing is complete
                logger.info("Embedding generator received termination signal, finishing")
                if isinstance(output, asyncio.Queue):
                    await output.put(None)  # Forward the termination signal
                break
            
            try:
                # Extract text for embedding
                text = self._prepare_text(document)
                
                # Generate embedding
                embedding = await self._generate_embedding(text)
                
                # Add embedding to the document
                document[self.embedding_key] = embedding
                
                self.processed_count += 1
                logger.info(f"Generated embedding for document {self.processed_count}: {document.get('title', 'Untitled')}")
                
                # Output the enhanced document
                if isinstance(output, asyncio.Queue):
                    await output.put(document)
                elif isinstance(output, dict):
                    output_key = document.get("url", f"doc_{self.processed_count}")
                    output[output_key] = document
                elif isinstance(output, io.TextIOWrapper):
                    output.write(json.dumps(document) + "\n")
                
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                logger.error(f"Call stack:\n{traceback.format_exc()}")
            
            # Mark the item as processed
            input.task_done()
    
    # Keep the old method name for backward compatibility
    async def generate_embeddings(self, input, output=None):
        """Legacy method, use embed() instead."""
        logger.warning("generate_embeddings() is deprecated, use embed() instead")
        return await self.embed(input, output)


class SentenceTransformerEmbeddingGenerator(BaseAsyncEmbeddingGenerator):
    """
    Embedding generator using SentenceTransformer models.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
        batch_size: int = 32,
        embedding_key: str = "embedding"
    ):
        super().__init__(embedding_key)
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None  # Lazy loading
    
    async def initialize_model(self):
        """Initialize the SentenceTransformer model asynchronously."""
        if self.model is None:
            try:
                logger.info(f"Initializing SentenceTransformer model: {self.model_name}")
                # Import here to allow for installation of the package if needed
                from sentence_transformers import SentenceTransformer
                # Run model initialization in a thread to not block the event loop
                self.model = await asyncio.to_thread(SentenceTransformer, self.model_name)
                logger.info("Model initialization complete")
            except Exception as e:
                logger.error(f"Error initializing SentenceTransformer model: {str(e)}")
                logger.error(f"Call stack:\n{traceback.format_exc()}")
                raise
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using SentenceTransformer."""
        if not text.strip():
            # Return a zero vector of appropriate size if text is empty
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        try:
            # Use the model to generate embeddings asynchronously
            embedding = await asyncio.to_thread(self.model.encode, text)
            
            # Convert numpy array to list for JSON serialization
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding with SentenceTransformer: {str(e)}")
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            # Return zero vector for failed embeddings
            return [0.0] * self.model.get_sentence_embedding_dimension()


class OllamaEmbeddingGenerator(BaseAsyncEmbeddingGenerator):
    """
    Embedding generator using Ollama API for embeddings.
    """
    
    def __init__(
        self, 
        model_name: str = "nomic-embed-text",  # or any embedding model available in Ollama
        api_host: str = "http://localhost:11434",
        embedding_key: str = "embedding"
    ):
        super().__init__(embedding_key)
        self.model_name = model_name
        self.api_host = api_host
        self.client = None
    
    async def initialize_model(self):
        """Initialize the Ollama client."""
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
        """Generate an embedding using Ollama API."""
        if not text.strip():
            return []  # Empty embedding for empty text
        
        try:
            # Call Ollama embeddings API
            response = await self.client.embeddings(
                model=self.model_name,
                prompt=text
            )
            
            # Extract embedding from response
            if hasattr(response, 'embedding'):
                return response.embedding
            elif isinstance(response, dict) and 'embedding' in response:
                return response['embedding']
            else:
                logger.error(f"Unexpected response format from Ollama: {response}")
                return []
            
        except Exception as e:
            logger.error(f"Error getting embedding from Ollama: {str(e)}")
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            return []
        
class OpenAIEmbeddingGenerator(BaseAsyncEmbeddingGenerator):
    """
    Embedding generator using OpenAI's API for embeddings.
    
    Processes documents asynchronously and generates embeddings using
    OpenAI's text-embedding models.
    """
    
    def __init__(
        self, 
        model_name: str = "text-embedding-3-large",
        api_key: str = None,
        dimensions: int = 3072,  # Update this to 3072 for full model capacity
        embedding_key: str = "$vector",
        max_text_length: int = 8191,  # OpenAI limit is 8191 tokens
    ):
        super().__init__(embedding_key)
        self.model_name = model_name
        self.dimensions = dimensions
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        self.max_text_length = max_text_length
    
    async def initialize_model(self):
        """Initialize the OpenAI client."""
        if self.client is None:
            try:
                logger.info(f"Initializing OpenAI client for model: {self.model_name}")
                from openai import AsyncOpenAI
                
                if not self.api_key:
                    raise ValueError("OpenAI API key not found. Set it in .env or pass it to the constructor.")
                
                self.client = AsyncOpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialization complete")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
                logger.error(f"Call stack:\n{traceback.format_exc()}")
                raise
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using OpenAI API."""
        if not text.strip():
            # Return a zero vector of appropriate size for empty text
            return [0.0] * self.dimensions
        
        try:
            # Optionally truncate if text is very long
            if len(text) > self.max_text_length * 4:  # Rough character estimate
                logger.warning(f"Text exceeds recommended length. Truncating from {len(text)} chars.")
                text = text[:self.max_text_length * 4]  # Rough truncation
            
            # Call OpenAI embeddings API
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimensions,
            )
            
            # Extract the embedding from the response
            if response and response.data and len(response.data) > 0:
                return response.data[0].embedding
            else:
                logger.error(f"Empty or unexpected response from OpenAI: {response}")
                return [0.0] * self.dimensions
            
        except Exception as e:
            logger.error(f"Error getting embedding from OpenAI: {str(e)}")
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            # Return zero vector for failed embeddings
            return [0.0] * self.dimensions