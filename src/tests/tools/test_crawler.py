import asyncio
import os
import tempfile
import pytest
from graphrag_agent.tools.crawler import AsyncFileSystemCrawler


@pytest.fixture
def sample_dir_structure(tmp_path):
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
    return tmp_path


@pytest.fixture
def nested_dir_structure(tmp_path):
    # a/
    #   f1.txt
    #   b/
    #     f2.txt
    #     c/
    #       f3.txt
    d1 = tmp_path / "a"
    d1.mkdir()
    d2 = d1 / "b"
    d2.mkdir()
    d3 = d2 / "c"
    d3.mkdir()
    (d1 / "f1.txt").write_text("a")
    (d2 / "f2.txt").write_text("b")
    (d3 / "f3.txt").write_text("c")
    return tmp_path


@pytest.fixture
def discard_dir_structure(tmp_path):
    # dir/
    #   skip.txt
    #   keep.txt
    d1 = tmp_path / "dir"
    d1.mkdir()
    (d1 / "skip.txt").write_text("skip me")
    (d1 / "keep.txt").write_text("keep me")
    return tmp_path


@pytest.mark.asyncio
async def test_filesystem_crawler_finds_files_with_extension(sample_dir_structure):
    tmp_path = sample_dir_structure
    crawler = AsyncFileSystemCrawler(str(tmp_path), max_depth=2, extensions=[".txt"])
    queue = asyncio.Queue()
    await crawler.run(depth=0, output=queue)

    found = set()
    while not queue.empty():
        item = await queue.get()
        if item is not None:
            found.add(os.path.basename(item))

    assert "file1.txt" in found
    assert "file3.txt" in found
    assert "file2.md" not in found


# @pytest.mark.asyncio
# async def test_filesystem_crawler_respects_max_depth(nested_dir_structure):
#     tmp_path = nested_dir_structure
#     crawler = AsyncFileSystemCrawler(str(tmp_path), max_depth=1, extensions=[".txt"])
#     queue = asyncio.Queue()
#     await crawler.run(str(tmp_path), 0, queue)

#     found = set()
#     while not queue.empty():
#         item = await queue.get()
#         if item is not None:
#             found.add(os.path.basename(item))

#     assert "f1.txt" in found
#     assert "f2.txt" not in found
#     assert "f3.txt" not in found


# @pytest.mark.asyncio
# async def test_filesystem_crawler_handles_permission_error(monkeypatch, tmp_path):
#     # Simulate PermissionError on scandir
#     def raise_permission_error(path):
#         raise PermissionError("Denied")

#     crawler = AsyncFileSystemCrawler(str(tmp_path), max_depth=1, extensions=[".txt"])
#     queue = asyncio.Queue()
#     monkeypatch.setattr(os, "scandir", raise_permission_error)
#     # Should not raise, just log warning
#     await crawler._run(str(tmp_path), 0, queue)


# @pytest.mark.asyncio
# async def test_filesystem_crawler_skips_discarded_files(discard_dir_structure):
#     tmp_path = discard_dir_structure
#     crawler = AsyncFileSystemCrawler(
#         str(tmp_path), max_depth=1, extensions=[".txt"], discard=["skip.txt"]
#     )

#     # Patch _should_include_file to skip files in discard list
#     def _should_include_file(self, filename):
#         return filename not in self.discard and any(
#             filename.endswith(ext) for ext in self.extensions
#         )

#     crawler._should_include_file = _should_include_file.__get__(crawler)

#     queue = asyncio.Queue()
#     await crawler._run(str(tmp_path), 0, queue)

#     found = set()
#     while not queue.empty():
#         item = await queue.get()
#         if item is not None:
#             found.add(os.path.basename(item))

#     assert "keep.txt" in found
#     assert "skip.txt" not in found


# def test_pytest_works():
#     assert True
