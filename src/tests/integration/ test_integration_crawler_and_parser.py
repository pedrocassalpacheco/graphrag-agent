import asyncio
import os
from pathlib import Path
import pytest

from graphrag_agent.tools.crawler import AsyncFileSystemCrawler
from graphrag_agent.tools.content_parser import AsyncMarkdownParser


@pytest.mark.asyncio
async def test_file_crawler_finds_python_files(tmp_path):
    # Setup: create a nested directory structure with .py and other files
    d1 = tmp_path / "dir1"
    d1.mkdir()
    (d1 / "a.py").write_text("print('a')")
    (d1 / "b.txt").write_text("not python")
    d2 = d1 / "subdir"
    d2.mkdir()
    (d2 / "c.py").write_text("print('c')")
    (d2 / "d.md").write_text("# markdown")

    crawler = AsyncFileSystemCrawler(
        base_path=str(tmp_path), max_depth=3, extensions=["py"]
    )
    queue = asyncio.Queue()
    await crawler.run(output=queue)

    found = set()
    while not queue.empty():
        item = await queue.get()
        if item is not None:
            found.add(os.path.basename(item))

    assert "a.py" in found
    assert "c.py" in found
    assert "b.txt" not in found
    assert "d.md" not in found


@pytest.mark.asyncio
async def test_markdown_parser_on_crawled_files(tmp_path):
    # Setup: create markdown files in a nested structure
    d1 = tmp_path / "docs"
    d1.mkdir()
    (d1 / "doc1.md").write_text("# Title\n\nSome content.")
    (d1 / "doc2.md").write_text("## Subtitle\n\nMore content.")
    (d1 / "not_markdown.txt").write_text("Just text.")

    crawler_queue = asyncio.Queue()
    parser_queue = asyncio.Queue()

    crawler = AsyncFileSystemCrawler(base_path=str(d1), max_depth=2, extensions=["md"])
    parser = AsyncMarkdownParser()

    crawler_task = asyncio.create_task(crawler.run(output=crawler_queue))
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )

    await asyncio.gather(crawler_task, parser_task)

    found_titles = set()
    while not parser_queue.empty():
        item = await parser_queue.get()
        if item is not None and isinstance(item, dict):
            found_titles.update(item.keys())

    # Should find at least one heading from the markdown files
    assert "Title" in found_titles or "Subtitle" in found_titles
