import asyncio
import os
from pathlib import Path
import pytest
from typing import Any
from graphrag_agent.tools.crawler import AsyncFileSystemCrawler, AsyncWebCrawler


@pytest.fixture
def sample_dir_structure(tmp_path: Path):
    # dir1/
    #   file1.txt
    #   file2.md
    #   dir2/
    #     file3.txt
    d1 = tmp_path / "dir1"
    d1.mkdir()
    f1 = d1 / "file1.txt"
    f1.write_text("hello")
    f2 = d1 / "file2.md"
    f2.write_text("world")
    d2 = d1 / "dir2"
    d2.mkdir()
    f3 = d2 / "file3.txt"
    f3.write_text("foo")
    return d1


@pytest.fixture
def nested_dir_structure(tmp_path: Path):
    # a/
    #   f1.txt
    #   b/
    #     f2.txt
    #     c/
    #       f3.txt
    a = tmp_path / "a"
    a.mkdir()
    b = a / "b"
    b.mkdir()
    c = b / "c"
    c.mkdir()
    (a / "f1.txt").write_text("a")
    (b / "f2.txt").write_text("b")
    (c / "f3.txt").write_text("c")
    return a


@pytest.fixture
def discard_dir_structure(tmp_path: Path):
    # dir/
    #   skip.txt
    #   keep.txt
    d1 = tmp_path / "dir"
    d1.mkdir()
    (d1 / "skip.txt").write_text("skip me")
    (d1 / "keep.txt").write_text("keep me")
    return d1


@pytest.mark.asyncio
async def test_filesystem_crawler_finds_files_with_extension(sample_dir_structure: Any):
    crawler = AsyncFileSystemCrawler(
        base_path=str(sample_dir_structure), max_depth=2, extensions=[".txt"]
    )
    queue = asyncio.Queue()
    await crawler.run(output=queue)

    found = set()
    while not queue.empty():
        item = await queue.get()
        if item is not None:
            found.add(os.path.basename(item))

    assert "file1.txt" in found
    assert "file3.txt" in found
    assert "file2.md" not in found


@pytest.mark.asyncio
async def test_filesystem_crawler_respects_max_depth(nested_dir_structure: Any):
    crawler = AsyncFileSystemCrawler(
        base_path=str(nested_dir_structure), max_depth=1, extensions=[".txt"]
    )

    queue = asyncio.Queue()
    await crawler.run(output=queue)

    found = set()
    while not queue.empty():
        item = await queue.get()
        if item is not None:
            found.add(os.path.basename(item))

    assert "f1.txt" in found
    assert "f2.txt" in found
    assert "f3.txt" not in found


@pytest.mark.asyncio
async def test_filesystem_crawler_handles_permission_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Simulate PermissionError on scandir
    def raise_permission_error(path):
        raise PermissionError("Denied")

    crawler = AsyncFileSystemCrawler(
        base_path=str(tmp_path), max_depth=1, extensions=[".txt"]
    )
    queue = asyncio.Queue()
    monkeypatch.setattr(os, "scandir", raise_permission_error)
    # Should not raise, just log warning
    await crawler.run(output=queue)


@pytest.mark.asyncio
async def test_filesystem_crawler_skips_discarded_files(discard_dir_structure: Any):
    crawler = AsyncFileSystemCrawler(
        base_path=str(discard_dir_structure), max_depth=1, extensions=[".txt"]
    )
    # Patch _should_include_file to skip files in discard list
    crawler.discard = ["skip.txt"]
    orig_should_include_file = crawler._should_include_file

    def _should_include_file(self, filename):
        return filename not in self.discard and orig_should_include_file(filename)

    crawler._should_include_file = _should_include_file.__get__(crawler)

    queue = asyncio.Queue()
    await crawler.run(output=queue)

    found = set()
    while not queue.empty():
        item = await queue.get()
        if item is not None:
            found.add(os.path.basename(item))

    assert "keep.txt" in found
    assert "skip.txt" not in found


@pytest.mark.asyncio
async def test_filesystem_crawler_output_to_list(sample_dir_structure: Any):
    crawler = AsyncFileSystemCrawler(
        base_path=str(sample_dir_structure), max_depth=2, extensions=[".txt"]
    )
    output_list = []
    await crawler.run(output=output_list)

    found = set(os.path.basename(item) for item in output_list)
    assert "file1.txt" in found
    assert "file3.txt" in found
    assert "file2.md" not in found


@pytest.mark.asyncio
async def test_filesystem_crawler_output_to_file(
    tmp_path: Path, sample_dir_structure: Any
):
    crawler = AsyncFileSystemCrawler(
        base_path=str(sample_dir_structure), max_depth=2, extensions=[".txt"]
    )
    output_file_path = tmp_path / "output.txt"
    with output_file_path.open("w") as f:
        await crawler.run(output=f)

    # Read the file and check contents
    with output_file_path.open("r") as f:
        lines = [line.strip() for line in f.readlines()]
    found = set(os.path.basename(line) for line in lines)
    assert "file1.txt" in found
    assert "file3.txt" in found
    assert "file2.md" not in found


# ------------------- Web Crawler Unit Test -------------------

import httpx
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_async_web_crawler_crawls_links(monkeypatch):
    # Mock HTML content with two links
    html_content = """
    <html>
        <body>
            <a href="http://test.com/page1">Page 1</a>
            <a href="http://test.com/page2">Page 2</a>
        </body>
    </html>
    """

    # Mock httpx.AsyncClient.get to return the above HTML
    class MockResponse:
        def __init__(self, text):
            self.text = text
            self.status_code = 200

        def raise_for_status(self):
            pass

    async def mock_get(url, *args, **kwargs):
        return MockResponse(html_content)

    with patch("httpx.AsyncClient.get", new=mock_get):
        crawler = AsyncWebCrawler(base_path="http://test.com", max_depth=1)
        queue = asyncio.Queue()
        await crawler.run(output=queue)

        found = set()
        while not queue.empty():
            item = await queue.get()
            if item is not None:
                found.add(item)

        # Should have crawled the base URL and both links
        assert "http://test.com" in found or "http://test.com/" in found
        assert "http://test.com/page1" in found
        assert "http://test.com/page2" in found
