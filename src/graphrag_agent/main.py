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
from graphrag_agent.tools.crawler import AsyncWebCrawler
from graphrag_agent.tools.content_parser import AsyncPageContentParser
from graphrag_agent.tools.qa_generator import AsyncQuestionGenerator3


async def main():
    base_url = "https://datastax.github.io/graph-rag/"
    crawler_queue = asyncio.Queue()
    parser_queue = asyncio.Queue()

    crawler = AsyncWebCrawler(
        base_url=base_url, max_depth=2, include_external=False, delay=1.0
    )
    parser = AsyncPageContentParser(delay=1.0)
    question_generator = AsyncQuestionGenerator3(model="llama3.3:latest")

    # Run crawler and parser concurrently
    with open("qanda.jsonl", "w") as output_file:
        crawler_task = asyncio.create_task(crawler.run(crawler_queue))
        parser_task = asyncio.create_task(
            parser.parse(crawler_queue, output=parser_queue)
        )
        qa_task = asyncio.create_task(
            question_generator.process_queue(parser_queue, output=output_file)
        )

        # Wait for both tasks to complete
        await asyncio.gather(crawler_task, parser_task, qa_task)


if __name__ == "__main__":
    asyncio.run(main())

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
