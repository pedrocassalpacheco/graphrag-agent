import asyncio
import io
import json
import os
from typing import Any, Dict, Union, List
import traceback


from astrapy.constants import VectorMetric
from astrapy.info import (
    CollectionDefinition,
)

import asyncio
import os
import traceback
from typing import Any, Optional, override

from graphrag_agent.tools.async_processor import BaseAsyncProcessor
from graphrag_agent.tools.parse_content import ParsedContent
from graphrag_agent.tools.async_processor import BaseAsyncProcessor
from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class AsyncAstraDBRepository(BaseAsyncProcessor):
    """A handler class for interacting with AstraDB asynchronously. This class provides methods to initialize
    a connection to AstraDB, write documents to a collection, and retrieve documents using vector similarity search.
    Attributes:
        collection_name (str): The name of the AstraDB collection to interact with.
        token (str): The authentication token for AstraDB.
        endpoint (str): The API endpoint for AstraDB.
        keyspace (str): The keyspace to use in AstraDB.
        vector_dimension (int): The dimension of the vector embeddings. Defaults to 768.
        embedding_key (str): The key in the document that contains the embedding vector. Defaults to "embedding".
        batch_size (int): The number of documents to process in a single batch. Defaults to 20.
        truncate_collection (bool): Whether to truncate the collection upon initialization. Defaults to False.
        astra_client (Optional[DataAPIClient]): The AstraDB client instance.
        collection (Optional[Collection]): The AstraDB collection instance.
        processed_count (int): The number of documents processed so far.
    Methods:
        initialize_client():
            Initializes the AstraDB client and connects to the specified collection.
        _prepare_document(document: Dict[str, Any]) -> Dict[str, Any]:
            Prepares a document for insertion into AstraDB by removing the embedding key and generating an ID if necessary.
        write(input: asyncio.Queue, output: Union[asyncio.Queue, Dict, io.TextIOWrapper] = None) -> None:
            Processes documents from an input queue and writes them to AstraDB in batches.
        retrieve(query: str, limit: int = 10, include_similarity: bool = True, filter_condition: Dict = None) -> List[Dict[str, Any]]:
            Retrieves documents from AstraDB using vector similarity search.
    """

    def __init__(
        self,
        collection_name: str,
        token: str,
        endpoint: str,
        keyspace: str,
        vector_dimension: int = 768,
        server_embedding_key: str = "$vectorize",
        client_embedding_key: str = "$vector",
        batch_size: int = 20,
        truncate_collection: bool = False,
        use_vectorize: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.collection_name = collection_name
        self.token = token
        self.endpoint = endpoint
        self.keyspace = keyspace
        self.vector_dimension = vector_dimension
        self.embedding_key = server_embedding_key
        self.client_embedding_key = client_embedding_key
        self.batch_size = batch_size
        self.astra_client = None
        self.collection = None
        self.processed_count = 0
        self.truncate_collection = truncate_collection
        self.use_vectorize = use_vectorize

    async def _get_client(self):
        """Initialize the AstraDB client."""
        if self.astra_client is None:
            try:
                # Import AstraDB client
                from astrapy import DataAPIClient

                # Instantiate the client - note this is not async
                self.astra_client = DataAPIClient()

                # Get the database - method is named differently in newer versions
                self.astradb_database = self.astra_client.get_database(
                    token=self.token, api_endpoint=self.endpoint, keyspace=self.keyspace
                )

                # Check if collection exists (this is not async)
                collections = self.astradb_database.list_collection_names()

                if self.collection_name in collections:
                    logger.debug(
                        f"Collection {self.collection_name} already exists, connecting to it"
                    )
                    self.collection = self.astradb_database.get_collection(
                        self.collection_name
                    )
                    if self.truncate_collection:
                        self.collection.delete_many({})
                else:
                    # This is too hard coded - should be passed as a parameter

                    if self.use_vectorize:
                        logger.debug(
                            f"Creating vectorize collection {self.collection_name}"
                        )
                        self.collection = self.astradb_database.create_collection(
                            self.collection_name,
                            definition=(
                                CollectionDefinition.builder()
                                .set_vector_service(
                                    provider="nvidia", model_name="NV-Embed-QA"
                                )
                                .set_vector_metric(VectorMetric.COSINE)
                                .build()
                            ),
                        )
                    else:
                        logger.debug(
                            f"Creating non-vectorize collection {self.collection_name}"
                        )
                        self.collection = self.astradb_database.create_collection(
                            self.collection_name,
                            definition=(
                                CollectionDefinition.builder()
                                .set_vector_metric(VectorMetric.COSINE)
                                .set_vector_dimension(self.vector_dimension)
                                .build()
                            ),
                        )

                logger.info(
                    f"Successfully connected to AstraDB collection: {self.collection_name}"
                )

                return self.collection

            except Exception as e:
                logger.error(f"Error initializing AstraDB client: {str(e)}")
                raise

    async def sample(self) -> Dict[str, Any]:
        """Sample the collection to check if it is empty."""
        await self._get_client()

        try:
            # Check if the collection is empty
            doc = self.collection.find_one({})
            return doc
        except Exception as e:
            logger.error(f"Error sampling collection: {str(e)}")
            raise

    def _prepare_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare document for insertion into AstraDB.

        This is a mess. Need to clean up the logic and make it more readable.
        """
        logger.debug(f"Preparing document for AstraDB: {str(document)[:100]}...")

        if document is None:
            return None

        # Process the content field if it exists
        if "content" in document:
            # Handle nested content structures
            if isinstance(document["content"], list):
                # Flatten any nested lists and convert everything to strings
                flattened_content = []
                for item in document["content"]:
                    if isinstance(item, list):
                        # Convert nested list to string representation
                        flattened_content.append(
                            "\n".join(str(subitem) for subitem in item)
                        )
                    elif isinstance(item, dict):
                        # For dictionary items (like formatted tables), convert to string
                        flattened_content.append(str(item))
                    else:
                        # Regular string items
                        flattened_content.append(str(item))

                # Join all content with newlines
                document["content"] = "\n".join(flattened_content)

            # Add the vectorize field for AstraDB's vector search
            if self.use_vectorize:
                document[self.embedding_key] = document["content"]
        else:
            logger.warning(
                f"Document has no 'content' field for vectorization: {document.get('_id', 'unknown_id')}"
            )

        logger.debug(
            f"Prepared document for AstraDB: {document.get('title', 'unknown_id')} size of content: {len(document.get('content', ''))}"
        )
        return document

    async def _process_item(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert a single document into AstraDB.
        """
        try:

            await self._get_client(self)
            doc = self._prepare_document(document)
            if doc is not None:
                result = await asyncio.to_thread(self.collection.insert_one, doc)
                self.processed_count += 1
                logger.info(f"Inserted document {doc.get('_id', 'unknown_id')}")
                return {"status": "written", "id": doc.get("_id", "unknown_id")}
            else:
                return {"status": "skipped", "id": document.get("_id", "unknown_id")}
        except Exception as e:
            logger.error(f"Error inserting document: {e}")
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            return {
                "status": "error",
                "id": document.get("_id", "unknown_id"),
                "error": str(e),
            }

    async def write(
        self,
        input: asyncio.Queue,
        output: Union[asyncio.Queue, Dict, io.TextIOWrapper] = None,
    ) -> None:
        """Process documents from queue and write to AstraDB."""
        batch = []
        await self._get_client()

        async def _write_batch(batch_docs):
            """Helper to write a batch of documents to the database."""
            if not batch_docs:
                return 0

            try:

                result = await asyncio.to_thread(
                    self.collection.insert_many, batch_docs
                )
                count = len(batch_docs)
                self.processed_count += len(batch_docs)
                logger.info(
                    f"Batch written: {count} documents (total: {self.processed_count})"
                )
                return count
            except Exception as e:
                logger.error(f"Error writing batch: {e}")
                return 0

        while True:
            document = await input.get()

            if document is None:  # End signal
                # Write remaining documents (without processing None)
                await _write_batch(batch)

                # Forward termination signal
                if isinstance(output, asyncio.Queue):
                    await output.put(None)

                logger.info(
                    f"Completed writing {self.processed_count} documents to AstraDB"
                )
                break

            try:
                # Prepare document
                document = self._prepare_document(document)

                # Only proceed if document preparation succeeded
                if document is not None:
                    batch.append(document)

                    # Write batch when full
                    if len(batch) >= self.batch_size:
                        await _write_batch(batch)
                        batch = []

                    # Output status if requested
                    if output:
                        status = {
                            "id": document.get("_id", "unknown"),
                            "status": "written",
                        }
                        if isinstance(output, asyncio.Queue):
                            await output.put(status)
                        elif isinstance(output, dict):
                            output[document.get("_id", "unknown")] = status
                        elif isinstance(output, io.TextIOWrapper):
                            output.write(json.dumps(status) + "\n")

            except Exception as e:
                logger.error(f"Error processing document: {e}")
                logger.error(f"Call stack:\n{traceback.format_exc()}")

            # Log progress
            logger.info(
                f"Processed document: {self.processed_count + len(batch)} (in progress)"
            )

            # Mark as done
            input.task_done()

    async def retrieve(
        self,
        query: str,
        embedding_vector: List[float] = None,
        limit: int = 10,
        include_similarity: bool = True,
        filter_condition: Dict = None,
    ):
        """
        Retrieve documents from AstraDB using vector similarity search.

        Args:
            query (str): The query text to vectorize and search with (used if use_vectorize=True)
            embedding_vector (List[float], optional): Pre-generated embedding vector (required if use_vectorize=False)
            limit (int, optional): Maximum number of documents to return. Defaults to 10.
            include_similarity (bool, optional): Include similarity scores in results. Defaults to True.
            filter_condition (Dict, optional): Additional filter conditions for the query.

        Returns:
            List[Dict[str, Any]]: List of matching documents
        """
        await self.initialize_client()

        try:
            # Prepare find parameters
            find_params = filter_condition or {}

            # Set up the sort parameter based on vectorize mode
            if self.use_vectorize:
                # Using AstraDB's vectorize service - pass the text query
                sort_param = {"$vectorize": query}
                logger.debug(f"Using built-in vectorize with query: {query[:30]}...")
            else:
                # Using pre-generated embeddings - make sure we have a vector
                if embedding_vector is None:
                    raise ValueError(
                        "embedding_vector must be provided when use_vectorize=False"
                    )

                sort_param = {"$vector": embedding_vector}
                logger.debug(
                    f"Using pre-generated embedding vector for query: {query[:30]}..."
                )

            # Execute the search
            cursor = self.collection.find(
                find_params,
                sort=sort_param,
                limit=limit,
                include_similarity=include_similarity,
            )

            # Convert cursor to list
            results = list(cursor)
            logger.info(
                f"Retrieved {len(results)} documents for query: {query[:30]}..."
            )

            # Log the first few results for debugging
            for i, doc in enumerate(results[:3]):
                title = doc.get("title", "No title")
                similarity = doc.get("$similarity", "No similarity")
                logger.debug(f"Result {i+1}: {title} (similarity: {similarity})")

            return results

        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            return []

    async def _cleanup(self):
        """Cleanup AstraDB client resources if needed."""
        try:
            if self.astra_client and hasattr(self.astra_client, "close"):
                await asyncio.to_thread(self.astra_client.close)
                logger.info("AstraDB client closed.")
        except Exception as e:
            logger.error(f"Error closing AstraDB client: {str(e)}")


logger = get_logger(__name__)


class AsyncFAISSVectorStoreProcessor(BaseAsyncProcessor):
    """FAISS in-memory vector store processor using LangChain."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model_name: str = "text-embedding-3-large",
        embedding_dimensions: int = 3072,
        output_dir="index",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model_name = embedding_model_name
        self.embedding_dimensions = embedding_dimensions
        self.embedding_model: Optional[Any] = None
        self.output_dir: str = output_dir
        self.vectorstore: FAISS = None
        self._initialized = False

    async def _initialize_store(self) -> None:
        """Initialize FAISS vector store and OpenAI embeddings."""
        if self._initialized:
            return

        try:
            logger.info(
                f"Initializing FAISS vector store with OpenAI embeddings: {self.embedding_model_name}"
            )

            # Import required libraries
            from langchain.vectorstores import FAISS
            from langchain.embeddings import OpenAIEmbeddings
            from langchain_core.documents import Document

            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter."
                )

            # Initialize OpenAI embeddings
            self.embedding_model = OpenAIEmbeddings(
                model=self.embedding_model_name,
            )

            # Create empty FAISS vector store with a dummy document
            dummy_doc = Document(
                page_content="Initialization document",
                metadata={"type": "dummy", "source": "init"},
            )

            # Initialize FAISS store
            self.vectorstore = await asyncio.to_thread(
                FAISS.from_documents, [dummy_doc], self.embedding_model
            )

            self._initialized = True
            logger.info("FAISS vector store initialization complete")

        except Exception as e:
            logger.error(f"Error initializing FAISS vector store: {str(e)}")
            logger.error(f"Call stack:\n{traceback.format_exc()}")
            raise

    @override
    async def _process_item(self, item: ParsedContent) -> ParsedContent:
        """Process a ParsedContent item by adding it to the FAISS vector store."""
        try:
            # Ensure store is initialized
            if not self._initialized:
                await self._initialize_store()

            # Convert ParsedContent to LangChain Document
            document = item.to_langchain_document()

            # Add document to FAISS store (run in thread to avoid blocking)
            await asyncio.to_thread(self.vectorstore.add_documents, [document])

            self.processed_count += 1
            logger.debug(
                f"Added document to FAISS {self.processed_count}: {item.source} - {item.section}"
            )

            # Return the original item (pass-through)
            return item

        except Exception as e:
            logger.error(f"Error adding to FAISS vector store: {str(e)}")
            logger.debug(f"Call stack:\n{traceback.format_exc()}")
            # Return original item on error
            return item

    async def _cleanup(self) -> None:
        """Cleanup FAISS vector store."""
        if self.vectorstore:
            logger.info("Cleaning up FAISS vector store")
            self.vectorstore.save_local(self.output_dir)
            self.vectorstore = None
            self.embedding_model = None
        await super()._cleanup()
