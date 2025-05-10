import asyncio
import hashlib
import io
import json
import os
from typing import Any, Dict, Union

from dotenv import load_dotenv

from astrapy.constants import VectorMetric
from astrapy.info import (
    CollectionDefinition,
    CollectionVectorOptions,
    VectorServiceOptions,
)

from graphrag_agent.utils.logging_config import get_logger
from graphrag_agent.utils.utils import print_pretty_json

logger = get_logger(__name__)



class AsyncAstraDBWriter:
    """
    Asynchronous tool to write documents with embeddings to AstraDB.
    """
    
    def __init__(
        self, 
        collection_name: str,
        token: str,
        endpoint: str,
        keyspace: str,
        vector_dimension: int = 768,
        embedding_key: str = "embedding",
        batch_size: int = 20,
    ):
        self.collection_name = collection_name
        self.token = token
        self.endpoint = endpoint
        self.keyspace = keyspace
        self.vector_dimension = vector_dimension
        self.embedding_key = embedding_key
        self.batch_size = batch_size
        self.astra_client = None
        self.collection = None
        self.processed_count = 0
    
    async def initialize_client(self):
        """Initialize the AstraDB client."""
        if self.astra_client is None:
            try:
                # Import AstraDB client
                from astrapy import DataAPIClient
                
                # Instantiate the client - note this is not async
                self.astra_client = DataAPIClient()
                
                # Get the database - method is named differently in newer versions
                self.astradb_database = self.astra_client.get_database(
                    token=self.token,
                    api_endpoint=self.endpoint,
                    keyspace=self.keyspace
                )
                
                # Check if collection exists (this is not async)
                collections = self.astradb_database.list_collection_names()
                
                if  self.collection_name in collections:
                    logger.info(f"Collection {self.collection_name} already exists, connecting to it")
                    self.collection = self.astradb_database.get_collection(self.collection_name)
                    #self.collection.delete_many({})
                else:
                    # This is too hard coded - should be passed as a parameter
                    logger.info(f"Creating collection {self.collection_name}")
                    self.collection = self.astradb_database.create_collection(
                        self.collection_name,
                        definition=(CollectionDefinition.builder()
                            .set_vector_service(provider="nvidia", model_name="NV-Embed-QA")
                            .set_vector_metric(VectorMetric.DOT_PRODUCT)
                            .build()
                        ),
                    )
                
                logger.info(f"Successfully connected to AstraDB collection: {self.collection_name}")
                
            except Exception as e:
                logger.error(f"Error initializing AstraDB client: {str(e)}")
                raise
            
    def _prepare_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare document for insertion into AstraDB."""
        # Copy document without embedding
        astra_doc = {k: v for k, v in document.items() if k != self.embedding_key}
        
        # Generate ID if not present
        # if "_id" not in astra_doc:
        #     url = astra_doc.get("url", "")
        #     title = astra_doc.get("title", "")
        #     #id_source = f"{url}:{title}" if url and title else str(astra_doc)
        #     #astra_doc["_id"] = hashlib.md5(id_source.encode()).hexdigest()
        
        # Add vector (must use this specific key name for AstraDB)
        # if self.embedding_key in document:
        #     astra_doc["$vector"] = document[self.embedding_key]
        
        return astra_doc
        
    async def write(
        self,
        input: asyncio.Queue,
        output: Union[asyncio.Queue, Dict, io.TextIOWrapper] = None,
    ) -> None:
        """Process documents from queue and write to AstraDB."""
        await self.initialize_client()
        
        batch = []
        total_documents = 0  # Track total documents processed
        
        while True:
            document = await input.get()
            print_pretty_json(document)
            import sys;sys
            if document is None:  # End signal
                # Write remaining documents
                if batch:
                    try:
                        result = await asyncio.to_thread(
                            self.collection.insert_many, batch
                        )
                        self.processed_count += len(batch)
                        total_documents += len(batch)
                        logger.info(f"Final batch written: {len(batch)} documents")
                    except Exception as e:
                        logger.error(f"Error writing final batch: {e}")
                
                # Forward termination signal
                if isinstance(output, asyncio.Queue):
                    await output.put(None)
                
                logger.info(f"Completed writing {self.processed_count} documents to AstraDB")
                logger.info(f"Total documents processed: {total_documents}")
                break
            
            try:
                # Prepare document
                astra_doc = self._prepare_document(document)
                batch.append(astra_doc)
                
                # Log document preparation
                logger.debug(f"Prepared document for AstraDB: {astra_doc.get('_id', 'unknown_id')}")
                
                # Write batch when full
                if len(batch) >= self.batch_size:
                    result = await asyncio.to_thread(
                        self.collection.insert_many, batch
                    )
                    self.processed_count += len(batch)
                    total_documents += len(batch)
                    logger.info(f"Batch written: {len(batch)} documents (total: {self.processed_count})")
                    batch = []
                
                # Output status if requested
                if output:
                    status = {"id": astra_doc["_id"], "status": "written"}
                    if isinstance(output, asyncio.Queue):
                        await output.put(status)
                    elif isinstance(output, dict):
                        output[astra_doc["_id"]] = status
                    elif isinstance(output, io.TextIOWrapper):
                        output.write(json.dumps(status) + "\n")
                
            except Exception as e:
                logger.error(f"Error processing document: {e}")
            
            # Log progress
            logger.info(f"Processed document: {self.processed_count + len(batch)} (in progress)")
            
            # Mark as done
            input.task_done()
            