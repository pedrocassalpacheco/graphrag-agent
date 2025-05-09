import asyncio
import json
import io
from typing import Dict, List, Union, Any, Optional
import numpy as np

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)

class BaseAsyncEmbeddingGenerator:
    """
    Base class for asynchronous embedding generators.
    
    This abstract class defines the interface for embedding generators that
    process documents from a queue and add embeddings to them.
    """
    
    def __init__(self, embedding_key: str = "embedding"):
        self.embedding_key = embedding_key
        self.processed_count = 0
    
    async def initialize_model(self):
        """Initialize the embedding model (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def _prepare_text(self, document: Dict[str, Any]) -> str:
        """Extract and prepare text from a document for embedding."""
        title = document.get("title", "")
        content = document.get("content", [])
        
        # Create a single string from title and content
        text_parts = [title]
        
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "image" in item:
                # Skip images or include alt text if needed
                alt = item["image"].get("alt", "")
                if alt:
                    text_parts.append(alt)
        
        return " ".join(text_parts)
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text (to be implemented by subclasses)."""
        raise NotImplementedError("Subclasses must implement this method")
    
    async def generate_embeddings(
        self,
        input: asyncio.Queue,
        output: Union[
            asyncio.Queue, Dict[str, Dict[str, Any]], io.TextIOWrapper
        ] = None,
    ) -> None:
        """Process documents from input queue, generate embeddings, and output results."""
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
            
            # Mark the item as processed
            input.task_done()


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
            logger.info(f"Initializing SentenceTransformer model: {self.model_name}")
            # Import here to allow for installation of the package if needed
            from sentence_transformers import SentenceTransformer
            # Run model initialization in a thread to not block the event loop
            self.model = await asyncio.to_thread(SentenceTransformer, self.model_name)
            logger.info("Model initialization complete")
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding using SentenceTransformer."""
        if not text.strip():
            # Return a zero vector of appropriate size if text is empty
            return [0.0] * self.model.get_sentence_embedding_dimension()
        
        # Use the model to generate embeddings asynchronously
        embedding = await asyncio.to_thread(self.model.encode, text)
        
        # Convert numpy array to list for JSON serialization
        return embedding.tolist()


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
            logger.info(f"Initializing Ollama client for model: {self.model_name}")
            from ollama import AsyncClient
            self.client = AsyncClient(host=self.api_host)
            logger.info("Ollama client initialization complete")
    
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
            return []