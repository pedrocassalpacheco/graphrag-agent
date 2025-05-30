import asyncio
import os
import pytest
from dotenv import load_dotenv
from typing import Dict, Any, List
from graphrag_agent.tools.vector_store import AsyncAstraDBRepository

pytestmark = pytest.mark.skip(reason="All crawler tests temporarily disabled")


@pytest.fixture(scope="session")
def env_config():
    """Load test environment configuration."""
    load_dotenv(".env.test")
    return {
        "endpoint": os.getenv("ASTRA_ENDPOINT"),
        "token": os.getenv("ASTRA_TOKEN"),
        "keyspace": os.getenv("ASTRA_KEYSPACE", "default_keyspace"),
    }


@pytest.fixture
async def astra_repo_non_vectorize(env_config):
    """Create AsyncAstraDBRepository instance with vectorize disabled."""
    repo = AsyncAstraDBRepository(
        collection_name="test_non_vectorize_collection",
        token=env_config["token"],
        endpoint=env_config["endpoint"],
        keyspace=env_config["keyspace"],
        use_vectorize=False,
        vector_dimension=384,
        truncate_collection=True,
    )
    await repo.initialize_client()
    yield repo
    await repo._cleanup()


@pytest.mark.asyncio
async def test_initialize_client_non_vectorize_mode(astra_repo_non_vectorize):
    """Test client initialization with non-vectorize mode."""
    assert astra_repo_non_vectorize.astra_client is not None
    assert astra_repo_non_vectorize.collection is not None
    assert astra_repo_non_vectorize.use_vectorize is False
    assert astra_repo_non_vectorize.vector_dimension == 384


@pytest.mark.asyncio
async def test_prepare_document_non_vectorize(astra_repo_non_vectorize):
    """Test document preparation in non-vectorize mode."""
    document = {
        "_id": "test_doc_non_vectorize",
        "title": "Test Non-Vectorize Document",
        "content": "This is content for non-vectorize mode",
    }

    prepared = astra_repo_non_vectorize._prepare_document(document)

    assert prepared["_id"] == "test_doc_non_vectorize"
    assert prepared["content"] == "This is content for non-vectorize mode"
    assert "$vectorize" not in prepared  # Should not have vectorize field


@pytest.mark.asyncio
async def test_retrieve_non_vectorize_requires_embedding():
    """Test that non-vectorize mode requires embedding vector for retrieval."""
    env_config = {
        "endpoint": os.getenv("ASTRA_ENDPOINT"),
        "token": os.getenv("ASTRA_TOKEN"),
        "keyspace": os.getenv("ASTRA_KEYSPACE", "default_keyspace"),
    }

    repo = AsyncAstraDBRepository(
        collection_name="test_non_vectorize_search",
        token=env_config["token"],
        endpoint=env_config["endpoint"],
        keyspace=env_config["keyspace"],
        use_vectorize=False,
        vector_dimension=384,
        truncate_collection=True,
    )

    await repo.initialize_client()

    # Should raise error when no embedding vector provided
    with pytest.raises(ValueError, match="embedding_vector must be provided"):
        await repo.retrieve(query="test query")

    await repo._cleanup()


@pytest.mark.asyncio
async def test_retrieve_non_vectorize_with_embedding(astra_repo_non_vectorize):
    """Test retrieval in non-vectorize mode with provided embedding vector."""
    # First insert a test document
    test_doc = {
        "_id": "search_doc_non_vectorize",
        "title": "Non-Vectorize Search Test",
        "content": "This is a test document for non-vectorize search",
    }

    await astra_repo_non_vectorize._process_item(test_doc)
    await asyncio.sleep(2)  # Wait for indexing

    # Create a dummy embedding vector (384 dimensions as configured)
    dummy_embedding = [0.1] * 384

    # Search should work with embedding vector provided
    results = await astra_repo_non_vectorize.retrieve(
        query="test search", embedding_vector=dummy_embedding, limit=5
    )

    # Should not raise error and return results as list
    assert isinstance(results, list)
    """Test client initialization with vectorize mode."""
    assert astra_repo_vectorize.astra_client is not None
    assert astra_repo_vectorize.collection is not None
    assert astra_repo_vectorize.use_vectorize is True


@pytest.mark.asyncio
async def test_initialize_client_non_vectorize_mode(astra_repo_non_vectorize):
    """Test client initialization with non-vectorize mode."""
    assert astra_repo_non_vectorize.astra_client is not None
    assert astra_repo_non_vectorize.collection is not None
    assert astra_repo_non_vectorize.use_vectorize is False


@pytest.mark.asyncio
async def test_sample_empty_collection(astra_repo_vectorize):
    """Test sampling an empty collection."""
    result = await astra_repo_vectorize.sample()
    assert result is None


@pytest.mark.asyncio
async def test_prepare_document_simple_content(astra_repo_vectorize):
    """Test document preparation with simple string content."""
    document = {
        "_id": "test_doc_1",
        "title": "Test Document",
        "content": "This is a test document content",
    }

    prepared = astra_repo_vectorize._prepare_document(document)

    assert prepared["_id"] == "test_doc_1"
    assert prepared["title"] == "Test Document"
    assert prepared["content"] == "This is a test document content"
    assert prepared["$vectorize"] == "This is a test document content"


@pytest.mark.asyncio
async def test_prepare_document_list_content(astra_repo_vectorize):
    """Test document preparation with list content."""
    document = {
        "_id": "test_doc_2",
        "title": "Test Document List",
        "content": ["First paragraph", "Second paragraph", "Third paragraph"],
    }

    prepared = astra_repo_vectorize._prepare_document(document)

    assert prepared["content"] == "First paragraph\nSecond paragraph\nThird paragraph"
    assert (
        prepared["$vectorize"] == "First paragraph\nSecond paragraph\nThird paragraph"
    )


@pytest.mark.asyncio
async def test_prepare_document_no_content(astra_repo_vectorize):
    """Test document preparation with no content field."""
    document = {"_id": "test_doc_3", "title": "Test Document No Content"}

    prepared = astra_repo_vectorize._prepare_document(document)

    assert prepared["_id"] == "test_doc_3"
    assert prepared["title"] == "Test Document No Content"
    assert "$vectorize" not in prepared


@pytest.mark.asyncio
async def test_prepare_document_none_input(astra_repo_vectorize):
    """Test document preparation with None input."""
    prepared = astra_repo_vectorize._prepare_document(None)
    assert prepared is None


@pytest.mark.asyncio
async def test_process_item_single_document(astra_repo_vectorize):
    """Test processing a single document."""
    document = {
        "_id": "test_single_doc",
        "title": "Single Test Document",
        "content": "This is a single test document",
    }

    result = await astra_repo_vectorize._process_item(document)

    assert result["status"] == "written"
    assert result["id"] == "test_single_doc"
    assert astra_repo_vectorize.processed_count == 1


@pytest.mark.asyncio
async def test_process_item_invalid_document(astra_repo_vectorize):
    """Test processing an invalid document."""
    result = await astra_repo_vectorize._process_item(None)

    assert result["status"] == "skipped"


@pytest.mark.asyncio
async def test_write_batch_processing(astra_repo_vectorize):
    """Test writing multiple documents through queue processing."""
    # Create test documents
    documents = [
        {
            "_id": f"batch_doc_{i}",
            "title": f"Batch Document {i}",
            "content": f"This is batch document number {i}",
        }
        for i in range(5)
    ]

    # Create input queue
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    # Add documents to queue
    for doc in documents:
        await input_queue.put(doc)

    # Add termination signal
    await input_queue.put(None)

    # Process documents
    await astra_repo_vectorize.write(input_queue, output_queue)

    # Verify processing
    assert astra_repo_vectorize.processed_count == 5


@pytest.mark.asyncio
async def test_retrieve_with_vectorize(astra_repo_vectorize):
    """Test document retrieval using vectorize mode."""
    # First, insert some test documents
    test_docs = [
        {
            "_id": "search_doc_1",
            "title": "Python Programming",
            "content": "Python is a high-level programming language",
        },
        {
            "_id": "search_doc_2",
            "title": "Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence",
        },
        {
            "_id": "search_doc_3",
            "title": "Data Science",
            "content": "Data science combines statistics and programming",
        },
    ]

    # Insert documents
    for doc in test_docs:
        await astra_repo_vectorize._process_item(doc)

    # Wait a moment for indexing
    await asyncio.sleep(2)

    # Search for documents
    results = await astra_repo_vectorize.retrieve(
        query="programming language", limit=2, include_similarity=True
    )

    assert len(results) <= 2
    assert all("$similarity" in doc for doc in results)


@pytest.mark.asyncio
async def test_retrieve_with_filter(astra_repo_vectorize):
    """Test document retrieval with filter conditions."""
    # Insert test documents with metadata
    test_docs = [
        {
            "_id": "filtered_doc_1",
            "title": "Python Basics",
            "content": "Introduction to Python programming",
            "category": "tutorial",
            "level": "beginner",
        },
        {
            "_id": "filtered_doc_2",
            "title": "Advanced Python",
            "content": "Advanced Python programming concepts",
            "category": "tutorial",
            "level": "advanced",
        },
    ]

    for doc in test_docs:
        await astra_repo_vectorize._process_item(doc)

    await asyncio.sleep(2)

    # Search with filter
    results = await astra_repo_vectorize.retrieve(
        query="Python programming", filter_condition={"level": "beginner"}, limit=5
    )

    assert all(doc.get("level") == "beginner" for doc in results)


@pytest.mark.asyncio
async def test_retrieve_empty_results(astra_repo_vectorize):
    """Test retrieval when no documents match."""
    results = await astra_repo_vectorize.retrieve(
        query="nonexistent query that should return nothing", limit=10
    )

    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_retrieve_with_non_vectorize_mode():
    """Test retrieval in non-vectorize mode with embedding vector."""
    # This test would need actual embedding vectors
    # For now, test that it raises appropriate error
    env_config = {
        "endpoint": os.getenv("ASTRA_ENDPOINT"),
        "token": os.getenv("ASTRA_TOKEN"),
        "keyspace": os.getenv("ASTRA_KEYSPACE", "default_keyspace"),
    }

    repo = AsyncAstraDBRepository(
        collection_name="test_non_vectorize_search",
        token=env_config["token"],
        endpoint=env_config["endpoint"],
        keyspace=env_config["keyspace"],
        use_vectorize=False,
        truncate_collection=True,
    )

    await repo.initialize_client()

    # Should raise error when no embedding vector provided
    with pytest.raises(ValueError, match="embedding_vector must be provided"):
        await repo.retrieve(query="test query")

    await repo._cleanup()


@pytest.mark.asyncio
async def test_multiple_initialization_calls(astra_repo_vectorize):
    """Test that multiple initialization calls don't cause issues."""
    # Client should already be initialized from fixture
    assert astra_repo_vectorize.astra_client is not None

    # Call initialize again
    await astra_repo_vectorize.initialize_client()

    # Should still work fine
    assert astra_repo_vectorize.astra_client is not None


@pytest.mark.asyncio
async def test_collection_truncation():
    """Test collection truncation functionality."""
    env_config = {
        "endpoint": os.getenv("ASTRA_ENDPOINT"),
        "token": os.getenv("ASTRA_TOKEN"),
        "keyspace": os.getenv("ASTRA_KEYSPACE", "default_keyspace"),
    }

    # First repo to add data
    repo1 = AsyncAstraDBRepository(
        collection_name="test_truncation_collection",
        token=env_config["token"],
        endpoint=env_config["endpoint"],
        keyspace=env_config["keyspace"],
        truncate_collection=False,
    )
    await repo1.initialize_client()

    # Add a document
    doc = {"_id": "truncation_test_doc", "content": "This document should be deleted"}
    await repo1._process_item(doc)

    # Verify document exists
    sample = await repo1.sample()
    assert sample is not None

    await repo1._cleanup()

    # Second repo with truncation enabled
    repo2 = AsyncAstraDBRepository(
        collection_name="test_truncation_collection",
        token=env_config["token"],
        endpoint=env_config["endpoint"],
        keyspace=env_config["keyspace"],
        truncate_collection=True,
    )
    await repo2.initialize_client()

    # Collection should be empty now
    sample = await repo2.sample()
    assert sample is None

    await repo2._cleanup()


@pytest.mark.asyncio
async def test_batch_size_configuration(env_config):
    """Test different batch size configurations."""
    repo = AsyncAstraDBRepository(
        collection_name="test_batch_size_collection",
        token=env_config["token"],
        endpoint=env_config["endpoint"],
        keyspace=env_config["keyspace"],
        batch_size=2,
        truncate_collection=True,
    )
    await repo.initialize_client()

    assert repo.batch_size == 2

    await repo._cleanup()


@pytest.mark.asyncio
async def test_processed_count_tracking(astra_repo_vectorize):
    """Test that processed count is accurately tracked."""
    initial_count = astra_repo_vectorize.processed_count

    documents = [
        {"_id": f"count_doc_{i}", "content": f"Document {i}"} for i in range(3)
    ]

    for doc in documents:
        await astra_repo_vectorize._process_item(doc)

    assert astra_repo_vectorize.processed_count == initial_count + 3
