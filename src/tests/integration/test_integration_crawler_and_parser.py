import asyncio
import os
import ast
from pathlib import Path
from typing import List, Dict, Any

import pytest

from graphrag_agent.tools.content_parser import (
    AsyncLangflowDocsMarkdownParser,
    AsyncPythonComponentParser,
    AsyncNoParser,
)
from graphrag_agent.tools.crawler import AsyncFileSystemCrawler
from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)

# Actual paths from your codebase
LANGFLOW_DOCS_PATH = "/Users/pedropacheco/Projects/dev/langflow.current/docs/docs"
LANGFLOW_COMPONENTS_PATH = "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/components"
LANGFLOW_SAMPLES_PATH = "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/initial_setup/starter_projects"

# Specific test files that we know exist
TEST_DOC_FILE = "/Users/pedropacheco/Projects/dev/langflow.current/docs/docs/Components/components-vector-stores.md"
TEST_COMPONENT_FILE = "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/components/amazon/amazon_bedrock_embedding.py"
TEST_SAMPLE_FILE = "/Users/pedropacheco/Projects/dev/langflow.current/src/backend/base/langflow/initial_setup/starter_projects/sequential_tasks_agent.py"


async def collect_queue_results(queue: asyncio.Queue, max_items: int = 10) -> List[Any]:
    """Helper to collect results from a queue with limit."""
    results = []
    count = 0
    while not queue.empty() and count < max_items:
        item = await queue.get()
        if item is not None:
            results.append(item)
            count += 1
    return results


async def run_parser_on_file(parser_class, file_path: str) -> List[Dict[str, Any]]:
    """Helper to run a parser on a single file."""
    if not Path(file_path).exists():
        pytest.skip(f"Test file not found: {file_path}")

    queue_in = asyncio.Queue()
    queue_out = asyncio.Queue()

    parser = parser_class()

    # Add file to input queue
    await queue_in.put(file_path)
    await queue_in.put(None)  # Termination signal

    # Run parser
    await parser.run(input=queue_in, output=queue_out)

    return await collect_queue_results(queue_out)


async def run_crawler_parser_pipeline(
    base_path: str, extensions: List[str], parser_class, max_files: int = 3
):
    """Helper to run crawler-parser pipeline with file limit."""
    if not Path(base_path).exists():
        pytest.skip(f"Test path not found: {base_path}")

    crawler_queue = asyncio.Queue()
    parser_queue = asyncio.Queue()

    crawler = AsyncFileSystemCrawler(
        base_path=base_path, max_depth=2, extensions=extensions
    )
    parser = parser_class()

    # Run crawler and parser concurrently
    crawler_task = asyncio.create_task(crawler.run(output=crawler_queue))
    parser_task = asyncio.create_task(
        parser.run(input=crawler_queue, output=parser_queue)
    )

    await asyncio.gather(crawler_task, parser_task)

    return await collect_queue_results(parser_queue, max_files)


@pytest.mark.asyncio
async def test_crawler_finds_python_components():
    """Test crawler finds Python component files."""
    if not Path(LANGFLOW_COMPONENTS_PATH).exists():
        pytest.skip(f"Components path not found: {LANGFLOW_COMPONENTS_PATH}")

    crawler = AsyncFileSystemCrawler(
        base_path=LANGFLOW_COMPONENTS_PATH, max_depth=2, extensions=["py"]
    )
    queue = asyncio.Queue()
    await crawler.run(output=queue)

    found_files = await collect_queue_results(queue, max_items=5)

    assert len(found_files) > 0
    assert all(str(f).endswith(".py") for f in found_files)
    logger.info(f"Found {len(found_files)} Python component files")


@pytest.mark.asyncio
async def test_crawler_finds_markdown_docs():
    """Test crawler finds markdown documentation files."""
    if not Path(LANGFLOW_DOCS_PATH).exists():
        pytest.skip(f"Docs path not found: {LANGFLOW_DOCS_PATH}")

    crawler = AsyncFileSystemCrawler(
        base_path=LANGFLOW_DOCS_PATH, max_depth=2, extensions=["md"]
    )
    queue = asyncio.Queue()
    await crawler.run(output=queue)

    found_files = await collect_queue_results(queue, max_items=5)

    assert len(found_files) > 0
    assert all(str(f).endswith(".md") for f in found_files)
    logger.info(f"Found {len(found_files)} markdown documentation files")


@pytest.mark.asyncio
async def test_langflow_docs_parser_on_actual_file():
    """Test LangFlow docs parser on actual documentation file."""
    results = await run_parser_on_file(AsyncLangflowDocsMarkdownParser, TEST_DOC_FILE)

    assert len(results) > 0
    assert isinstance(results[0], dict)

    # Check that we got meaningful content
    content_found = False
    for result in results:
        if isinstance(result, dict):
            for value in result.values():
                if isinstance(value, (str, list)) and len(str(value)) > 50:
                    content_found = True
                    break

    assert content_found, "Should find substantial content in documentation"
    logger.info(f"Parsed {len(results)} sections from documentation")


@pytest.mark.asyncio
async def test_python_component_parser_on_actual_file():
    """Test Python component parser on actual component file."""
    results = await run_parser_on_file(AsyncPythonComponentParser, TEST_COMPONENT_FILE)

    assert len(results) > 0
    assert isinstance(results[0], dict)

    # Look for class or component information
    found_component_info = False
    for result in results:
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, dict) and "class_name" in value:
                    found_component_info = True
                    logger.info(f"Found component: {value.get('class_name')}")
                    break

    assert found_component_info, "Should find component class information"
    logger.info(f"Parsed {len(results)} items from component file")


@pytest.mark.asyncio
async def test_python_sample_parser_on_actual_file():
    """Test AsyncNoParser on actual sample file and validate Python syntax."""
    results = await run_parser_on_file(AsyncNoParser, TEST_SAMPLE_FILE)

    assert len(results) > 0
    assert isinstance(results[0], dict)

    # AsyncNoParser should return raw file content - validate it's valid Python
    found_valid_python = False

    for result in results:
        if isinstance(result, dict):
            content = result.get("content", "")
            file_path = result.get("file_path", "")

            if content and isinstance(content, str):
                try:
                    # Parse the content with AST to validate Python syntax
                    parsed_ast = ast.parse(content)

                    # Verify it's a valid AST with some content
                    assert parsed_ast is not None
                    assert isinstance(parsed_ast, ast.Module)
                    assert len(parsed_ast.body) > 0, "Should have actual code content"

                    found_valid_python = True
                    logger.info(
                        f"✅ Successfully validated Python syntax in {file_path}"
                    )
                    logger.info(
                        f"AST contains {len(parsed_ast.body)} top-level statements"
                    )

                    # Log what we found for debugging
                    functions = [
                        n
                        for n in ast.walk(parsed_ast)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    classes = [
                        n for n in ast.walk(parsed_ast) if isinstance(n, ast.ClassDef)
                    ]
                    logger.info(
                        f"Content analysis: {len(functions)} functions, {len(classes)} classes"
                    )

                    break

                except SyntaxError as e:
                    pytest.fail(
                        f"Invalid Python syntax in content from {file_path}: {e}"
                    )
                except Exception as e:
                    logger.error(f"Error parsing content with AST: {e}")
                    continue

    assert found_valid_python, "Should find valid Python content that parses with AST"
    logger.info("✅ AsyncNoParser successfully preserved valid Python syntax")


@pytest.mark.asyncio
async def test_docs_crawler_parser_pipeline():
    """Test complete pipeline: crawl docs → parse markdown."""
    results = await run_crawler_parser_pipeline(
        LANGFLOW_DOCS_PATH, ["md"], AsyncLangflowDocsMarkdownParser, max_files=2
    )

    assert len(results) > 0
    assert all(isinstance(r, dict) for r in results)

    total_content_length = sum(
        len(str(v))
        for result in results
        for v in result.values()
        if isinstance(v, (str, list))
    )
    assert total_content_length > 100, "Should extract substantial content"
    logger.info(f"Documentation pipeline processed {len(results)} sections")


@pytest.mark.asyncio
async def test_components_crawler_parser_pipeline():
    """Test complete pipeline: crawl components → parse Python."""
    results = await run_crawler_parser_pipeline(
        LANGFLOW_COMPONENTS_PATH, ["py"], AsyncPythonComponentParser, max_files=2
    )

    assert len(results) > 0
    assert all(isinstance(r, dict) for r in results)

    # Count classes found
    classes_found = 0
    for result in results:
        for key, value in result.items():
            if isinstance(value, dict) and "class_name" in value:
                classes_found += 1

    assert classes_found > 0, "Should find component classes"
    logger.info(f"Components pipeline found {classes_found} component classes")


@pytest.mark.asyncio
async def test_samples_crawler_parser_pipeline():
    """Test complete pipeline: crawl samples → parse Python functions."""
    results = await run_crawler_parser_pipeline(
        LANGFLOW_SAMPLES_PATH, ["py"], AsyncNoParser, max_files=2
    )

    assert len(results) > 0
    assert all(isinstance(r, dict) for r in results)

    # Count functions found by parsing the content with AST
    functions_found = 0
    valid_files = 0

    for result in results:
        if isinstance(result, dict):
            content = result.get("content", "")
            file_path = result.get("file_path", "unknown")

            if content and isinstance(content, str):
                try:
                    # Parse the content with AST to count functions
                    parsed_ast = ast.parse(content)

                    # Count function definitions
                    file_functions = [
                        node
                        for node in ast.walk(parsed_ast)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]

                    functions_found += len(file_functions)
                    valid_files += 1

                    logger.info(
                        f"File {file_path}: found {len(file_functions)} functions"
                    )

                except SyntaxError as e:
                    logger.warning(f"Syntax error in {file_path}: {e}")
                except Exception as e:
                    logger.warning(f"Error parsing {file_path}: {e}")

    assert valid_files > 0, "Should have at least one valid Python file"
    assert functions_found > 0, "Should find function definitions in Python files"
    logger.info(
        f"Samples pipeline found {functions_found} functions across {valid_files} files"
    )


@pytest.mark.asyncio
async def test_file_crawler_with_synthetic_data(tmp_path):
    """Test file crawler with controlled synthetic data."""
    # Create nested directory structure
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

    found_files = await collect_queue_results(queue)
    found_basenames = {os.path.basename(f) for f in found_files}

    assert "a.py" in found_basenames
    assert "c.py" in found_basenames
    assert "b.txt" not in found_basenames
    assert "d.md" not in found_basenames


@pytest.mark.asyncio
async def test_docs_parser():
    """Test AsyncLangflowDocsMarkdownParser processes markdown files correctly."""
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

    # Wait for parser task to complete
    await asyncio.gather(parser_task)

    # Assert parser produced results
    assert not parser_queue.empty(), "Parser queue should not be empty"

    # Get all results from queue
    results = []
    while not parser_queue.empty():
        results.append(parser_queue.get_nowait())

    assert len(results) > 0, "Should have at least one parsed result"
    assert isinstance(results[0], dict), "Results should be dictionaries"


@pytest.mark.asyncio
async def test_components_parser():
    """Test AsyncPythonComponentParser processes Python component files correctly."""
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

    # Wait for parser task to complete
    await asyncio.gather(parser_task)

    # Assert parser produced results
    assert not parser_queue.empty(), "Parser queue should not be empty"

    # Get all results from queue
    results = []
    while not parser_queue.empty():
        results.append(parser_queue.get_nowait())

    assert len(results) > 0, "Should have at least one parsed result"
    assert isinstance(results[0], dict), "Results should be dictionaries"


@pytest.mark.asyncio
async def test_examples_parser():
    """Test AsyncNoParser processes Python example files correctly."""
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

    # Wait for parser task to complete
    await asyncio.gather(parser_task)

    # Assert parser produced results
    assert not parser_queue.empty(), "Parser queue should not be empty"

    # Get all results from queue
    results = []
    while not parser_queue.empty():
        results.append(parser_queue.get_nowait())

    assert len(results) > 0, "Should have at least one parsed result"
    assert isinstance(results[0], dict), "Results should be dictionaries"


@pytest.mark.asyncio
async def test_samples_parser():
    """Test AsyncPythonComponentParser processes sample Python files correctly."""
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

    # Assert parser produced results
    assert not parser_queue.empty(), "Parser queue should not be empty"

    # Get all results from queue
    results = []
    while not parser_queue.empty():
        results.append(parser_queue.get_nowait())

    assert len(results) > 0, "Should have at least one parsed result"
    assert isinstance(results[0], dict), "Results should be dictionaries"
