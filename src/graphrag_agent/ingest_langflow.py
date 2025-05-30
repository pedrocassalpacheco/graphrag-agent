import asyncio
from graphrag_agent.tools.content_parser import (
    AsyncLangflowDocsMarkdownParser,
    AsyncPythonComponentParser,
    AsyncPythonExampleParser,
    AsyncNoParser,
)
from graphrag_agent.tools.crawler import AsyncFileSystemCrawler
from graphrag_agent.tools.content_parser import AsyncPythonComponentParser
from graphrag_agent.tools.document_embedding import OpenAIEmbeddingProcessor
from graphrag_agent.tools.vector_store import AsyncAstraDBRepository
from graphrag_agent.utils.utils import dump_queue
import os
from dotenv import load_dotenv

# Constants
load_dotenv()

ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = "langflowtest"  # os.getenv("ASTRA_DB_KEYSPACE", "langflow")
ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "langflow_docs")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY", "embedding")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def test_docs():
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    parser = AsyncLangflowDocsMarkdownParser()
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )

    crawler_queue.put_nowait(
        "/Users/pedropacheco/Projects/dev/langflow.current/docs/docs/Components/components-vector-stores.md"
    )
    crawler_queue.put_nowait(None)
    # Wait for both tasks to complete
    await asyncio.gather(parser_task)

    dump_queue(parser_queue)


async def test_components():
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    parser = AsyncPythonComponentParser()
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )

    crawler_queue.put_nowait(
        "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/components/amazon/amazon_bedrock_embedding.py"
    )
    crawler_queue.put_nowait(None)
    # Wait for both tasks to complete
    await asyncio.gather(parser_task)

    dump_queue(parser_queue)


async def test_examples():
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    parser = AsyncNoParser()
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )

    crawler_queue.put_nowait(
        "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/initial_setup/starter_projects/sequential_tasks_agent.py"
    )
    crawler_queue.put_nowait(None)
    # Wait for both tasks to complete
    await asyncio.gather(parser_task)

    dump_queue(parser_queue)


async def test_samples():
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    parser = AsyncPythonComponentParser()
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )

    crawler_queue.put_nowait(
        "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/initial_setup/starter_projects/sequential_tasks_agent.py"
    )
    crawler_queue.put_nowait(None)
    await asyncio.gather(parser_task)

    dump_queue(parser_queue)


async def ingest_docs():

    path = "/Users/pedropacheco/Projects/dev/langflow.current/docs/docs/Components"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    embedding_queue: asyncio.queues.Queue = asyncio.Queue()

    crawler = AsyncFileSystemCrawler(base_path=path, max_depth=3, extensions=["md"])
    parser = AsyncLangflowDocsMarkdownParser()
    embedding = OpenAIEmbeddingProcessor()
    vector_db = AsyncAstraDBRepository(
        collection_name="langflow_docs",
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=3072,
        truncate_collection=True,
        use_vectorize=False,
    )

    crawler_task = asyncio.create_task(crawler.run(output=crawler_queue))
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )
    emdedding_task = asyncio.create_task(
        embedding.run(input=parser_queue, output=embedding_queue)
    )
    # writer_task = asyncio.create_task(vector_db.write(input=embedding_queue))

    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task, emdedding_task)


async def ingest_components():
    path = "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/components"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    embedding_queue: asyncio.queues.Queue = asyncio.Queue()

    crawler = AsyncFileSystemCrawler(base_path=path, max_depth=3, extensions=["py"])
    parser = AsyncPythonComponentParser()
    vector_db = AsyncAstraDBRepository(
        collection_name="component_code",
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=3072,
        truncate_collection=True,
        use_vectorize=False,
    )
    embedding = OpenAIEmbeddingProcessor()

    # Fix: use output= for crawler.run()
    crawler_task = asyncio.create_task(crawler.run(output=crawler_queue))
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )
    emdedding_task = asyncio.create_task(
        embedding.run(input=parser_queue, output=embedding_queue)
    )
    writer_task = asyncio.create_task(vector_db.write(input=embedding_queue))

    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task, emdedding_task, writer_task)


async def ingest_samples():
    path = "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/initial_setup/starter_projects"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    embedding_queue: asyncio.queues.Queue = asyncio.Queue()

    # These are the components that will be used to crawl, parse, embed and write the data. Should be part of the agent, but for now just running it here.
    crawler = AsyncFileSystemCrawler(base_path=path, max_depth=3, extensions=["py"])
    parser = AsyncPythonSampleParser()

    embedding = OpenAIEmbeddingProcessor()

    vector_db = AsyncAstraDBRepository(
        collection_name="sample_code",
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=3072,
        truncate_collection=True,
        use_vectorize=False,
    )

    # Fix: use output= for crawler.run() and run() for parser
    crawler_task = asyncio.create_task(crawler.run(output=crawler_queue))
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )
    emdedding_task = asyncio.create_task(
        embedding.run(input=parser_queue, output=embedding_queue)
    )
    writer_task = asyncio.create_task(vector_db.write(input=embedding_queue))

    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task, emdedding_task, writer_task)


if __name__ == "__main__":
    asyncio.run(test_docs())
    asyncio.run(test_components())
    asyncio.run(test_examples())
