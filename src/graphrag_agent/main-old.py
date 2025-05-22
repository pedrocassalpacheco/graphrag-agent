import asyncio
from graphrag_agent.tools.crawler import AsyncWebCrawler, AsyncFileSystemCrawler
from graphrag_agent.tools.content_parser import AsyncPageContentParser
from graphrag_agent.tools.qa_generator import AsyncQuestionGenerator3
from graphrag_agent.tools.content_parser import AsyncMarkdownParser
from graphrag_agent.tools.document_embedding import OllamaEmbeddingGenerator
from graphrag_agent.tools.document_embedding import OpenAIEmbeddingGenerator
from graphrag_agent.tools.vector_store import AsyncAstraDBRepository
from graphrag_agent.tools.content_parser import AsyncPythonSourceParser
from graphrag_agent.tools.content_parser import AsyncLangflowDocsMarkdownParser
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from graphrag_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Constants
load_dotenv()

ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "langflow")
ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "langflow_docs")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY", "embedding")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


async def graph_rag():
    base_url = "https://datastax.github.io/graph-rag/"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()

    crawler = AsyncWebCrawler(base_url=base_url, max_depth=2, include_external=False)
    parser = AsyncPageContentParser(delay=1.0)
    qa_generator = AsyncQuestionGenerator3(model="mistral")

    # Run crawler and parser concurrently
    fp = open("output.jsonl", "w")
    await crawler.run(crawler_queue)
    crawler_task = asyncio.create_task(crawler.run(crawler_queue))
    parser_task = asyncio.create_task(
        parser.parse(input=crawler_queue, output=parser_queue)
    )
    qa_task = asyncio.create_task(
        qa_generator.generate_qa(input=parser_queue, output=fp)
    )
    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task, qa_task)

    while not parser_queue.empty():
        item = await parser_queue.get()
        print(item)


async def documentation_parsing():

    path = "/Users/pedropacheco/Projects/dev//langflow.current/docs/docs/Components"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    embedding_queue: asyncio.queues.Queue = asyncio.Queue()

    crawler = AsyncFileSystemCrawler(base_path=path, max_depth=3, extensions=["md"])
    parser = AsyncLangflowDocsMarkdownParser()
    vector_db = AsyncAstraDBRepository(
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=3072,
        truncate_collection=True,
        use_vectorize=False,
    )
    embedding = OpenAIEmbeddingGenerator(
        api_key=OPENAI_API_KEY,
    )

    crawler_task = asyncio.create_task(crawler.run(crawler_queue))
    parser_task = asyncio.create_task(
        parser.parse(input=crawler_queue, output=parser_queue)
    )
    emdedding_task = asyncio.create_task(
        embedding.embed(input=parser_queue, output=embedding_queue)
    )
    writer_task = asyncio.create_task(vector_db.write(input=embedding_queue))

    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task, emdedding_task, writer_task)

    # Then retrieve using the embedding vector
    query = "what are the inputs and outputs for the astrad db vector store component?"
    embedding_vector = await embedding._generate_embedding(query)

    # Use the generated vector for retrieval
    results = await vector_db.retrieve(
        query=query,
        embedding_vector=embedding_vector,  # Pass the actual vector, not the generator
        limit=10,
    )

    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Title: {doc.get('title', 'No title')}")
        print(f"Similarity: {doc.get('$similarity', 'No similarity')}")
        print("-" * 40)

    return results


async def component_flow():

    path = "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/components"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    embedding_queue: asyncio.queues.Queue = asyncio.Queue()

    crawler = AsyncFileSystemCrawler(base_path=path, max_depth=3, extensions=["py"])
    parser = AsyncPythonSourceParser()
    vector_db = AsyncAstraDBRepository(
        collection_name="component_code",
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=3072,
        truncate_collection=True,
        use_vectorize=False,
    )
    embedding = OpenAIEmbeddingGenerator(
        api_key=OPENAI_API_KEY,
    )

    crawler_task = asyncio.create_task(crawler.run(crawler_queue))
    parser_task = asyncio.create_task(
        parser.parse(input=crawler_queue, output=parser_queue)
    )
    emdedding_task = asyncio.create_task(
        embedding.embed(input=parser_queue, output=embedding_queue)
    )
    writer_task = asyncio.create_task(vector_db.write(input=embedding_queue))

    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task, emdedding_task, writer_task)


async def code_parsing():

    path = "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/components"

    # These are uses to pass the data between the different tools
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()
    vector_queue: asyncio.queues.Queue = asyncio.Queue()

    # Tools themselves
    crawler = AsyncFileSystemCrawler(
        base_path=path, max_depth=3, extensions=[".py"], discard=["__init__.py"]
    )
    parser = AsyncPythonSourceParser()
    vectorizer = OllamaEmbeddingGenerator()
    writer = AsyncAstraDBRepository(
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT,
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=VECTOR_DIMENSION,
        truncate_collection=True,
    )

    # Tasks to run concurrently
    crawler_task = asyncio.create_task(crawler.run(crawler_queue))
    parser_task = asyncio.create_task(
        parser.parse(input=crawler_queue, output=parser_queue)
    )
    vector_task = asyncio.create_task(
        vectorizer.embed(input=parser_queue, output=vector_queue)
    )
    writer_task = asyncio.create_task(writer.write(input=vector_queue))

    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task, vector_task, writer_task)


def dump_queue(queue: asyncio.Queue):
    """
    Dump the contents of a queue to a file.
    """
    with open("queue_dump.txt", "w") as f:
        while not queue.empty():
            item = queue.get_nowait()
            f.write(f"{item}\n")


if __name__ == "__main__":

    async def run(self, queue: asyncio.Queue):
        logger.info(f"Starting file scan from {self.base_path}")
        await self._run(self.base_path, 0, queue)
        await queue.put(None)
        logger.info(f"File scan completed. Found {len(self.found_resources)} files")
        return self.found_resources
