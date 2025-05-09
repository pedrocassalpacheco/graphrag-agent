# import os
# import asyncio
# import torch
# import sys

# def main():
#     print("Hello from graphrag-agent!")


#     print(torch.cuda.is_available())
#     print(torch.cuda.device_count())
#     print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")


#     print(sys.executable)
#     print(sys.version)

#     from tools.async_crawler import AsyncWebCrawler

#     async_crawler = AsyncWebCrawler(base_url="https://datastax.github.io/graph-rag/", max_depth=10, include_external=False, delay=1.0)

#     results = asyncio.run(async_crawler.run())
#     from pprint import pprint  # Import pprint for pretty-printing
#     pprint(results)

# if __name__ == "__main__":
#     main()
import asyncio
from graphrag_agent.tools.crawler import AsyncWebCrawler, AsyncFileSystemCrawler
from graphrag_agent.tools.content_parser import AsyncPageContentParser
from graphrag_agent.tools.qa_generator import AsyncQuestionGenerator3
from graphrag_agent.tools.content_parser import AsyncMarkdownParser
from graphrag_agent.tools.document_embedding import OllamaEmbeddingGenerator
import os
from dotenv import load_dotenv

# Constants
load_dotenv()
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT") 
ASTRA_DB_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "langflow")
ASTRA_DB_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "langflow_docs")
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "768"))
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY", "embedding")

async def graph_rag():
    base_url = "https://datastax.github.io/graph-rag/"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()

    crawler = AsyncWebCrawler(
        base_url=base_url, max_depth=2, include_external=False, delay=1.0
    )
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
    await asyncio.gather(crawler_task, parser_task,qa_task)
    
    while not parser_queue.empty():
        item = await parser_queue.get()
        print(item)


async def prompt_flow():
    
    path = "/opt/langflow/docs/docs/components"
    crawler_queue: asyncio.queues.Queue = asyncio.Queue()
    parser_queue: asyncio.queues.Queue = asyncio.Queue()

    crawler = AsyncFileSystemCrawler(
        base_path=path, max_depth=3, extensions=["md"], delay=1.0
    )
    parser = AsyncMarkdownParser(delay=1.0)
    embedding = OllamaEmbeddingGenerator(model_name="nomic-embed-text")

    crawler_task = asyncio.create_task(crawler.run(crawler_queue))
    parser_task = asyncio.create_task(
         parser.parse(input=crawler_queue, output=parser_queue)
    )
    fp = open("embeddings.jsonl", "w")
    embedding_task = asyncio.create_task(
         embedding.generate_embeddings(input=parser_queue, output=fp)
    )

    # Wait for both tasks to complete
    await asyncio.gather(crawler_task, parser_task, embedding_task)
    
    fp.close()
        
async def db():
    # Initialize the database connection
    from graphrag_agent.tools.vector_store import AsyncAstraDBWriter
    vector_store = AsyncAstraDBWriter(
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_TOKEN,
        endpoint=ASTRA_DB_ENDPOINT, 
        keyspace=ASTRA_DB_KEYSPACE,
        vector_dimension=VECTOR_DIMENSION
    )
    
    await vector_store.initialize_client()
    # Perform operations with the vector store
    # For example, inserting documents, querying, etc.
    pass
        
if __name__ == "__main__":
    asyncio.run(db())
    #asyncio.run(file_crawler())
    # # Save results to a JSON Lines file
    # import json
    # with open("results.jsonl", "w") as jsonl_file:
    #     for url, content in results.items():
    #         for title, paragraphs in content.items():
    #             json_line = {
    #                 "url": url,
    #                 "title": title,
    #                 "content": paragraphs
    #             }
    #             jsonl_file.write(json.dumps(json_line) + "\n")

    #asyncio.run(file_crawler())