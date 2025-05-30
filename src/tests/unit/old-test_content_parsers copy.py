import os
import pytest
from pathlib import Path
from dotenv import load_dotenv

from graphrag_agent.tools.content_parser import (
    AsyncPageContentParser,
    AsyncMarkdownParser,
    AsyncLangflowDocsMarkdownParser,
    AsyncPythonComponentParser,
    AsyncPythonSampleParser,
)
from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)

# Load test-specific environment variables
load_dotenv(dotenv_path=".env.test")

# Test paths from environment
MARKDOWN_DOCS_PATH = os.getenv("MARKDOWN_DOCS_PATH")
COMPONENT_CODE_PATH = os.getenv("COMPONENT_CODE_PATH")
SAMPLE_CODE_PATH = os.getenv("SAMPLE_CODE_PATH")


@pytest.mark.asyncio
async def test_async_markdown_parser_on_file(tmp_path):
    """Test AsyncMarkdownParser parses a markdown file correctly."""
    test_md = tmp_path / "test.md"
    test_md.write_text("# Title\n\nSome content.\n\n## Subtitle\nMore content.")

    parser = AsyncMarkdownParser()
    results = []
    async for result in parser._process_item(str(test_md)):
        results.append(result)

    assert len(results) > 0
    assert any(isinstance(result, dict) for result in results)
    # Check if any result contains our expected content
    found_content = False
    for result in results:
        if isinstance(result, dict):
            for key, value in result.items():
                if "Title" in key or "Subtitle" in key:
                    found_content = True
                    break
    assert found_content


@pytest.mark.asyncio
async def test_async_markdown_parser_on_nonexistent_file(tmp_path):
    """Test AsyncMarkdownParser handles nonexistent file gracefully."""
    parser = AsyncMarkdownParser()
    results = []
    async for result in parser._process_item(str(tmp_path / "does_not_exist.md")):
        results.append(result)

    # Should not yield any results for nonexistent file
    assert len(results) == 0


@pytest.mark.asyncio
async def test_async_page_content_parser_on_html(httpserver):
    """Test AsyncPageContentParser parses HTML from a mocked HTTP server."""
    html_content = "<h1>Header</h1><p>Paragraph</p>"
    httpserver.expect_request("/test.html").respond_with_data(
        html_content, content_type="text/html"
    )

    url = httpserver.url_for("/test.html")
    parser = AsyncPageContentParser()
    results = []
    async for result in parser._process_item(url):
        results.append(result)

    logger.debug(f"Results: {results}")
    assert len(results) > 0
    assert isinstance(results[0], dict)
    # Check if the result contains expected content
    result = results[0]
    assert any("Header" in str(k) or "Paragraph" in str(v) for k, v in result.items())


@pytest.mark.asyncio
async def test_async_langflow_docs_markdown_parser(tmp_path):
    """Test AsyncLangflowDocsMarkdownParser parses a markdown file."""
    test_md = tmp_path / "synthetic_component.md"
    test_md.write_text(
        """## SyntheticMemoryComponent

This component creates a `SyntheticMemoryHistory` instance, enabling storage and retrieval of chat messages using a synthetic in-memory store for testing purposes.

<details>
<summary>Parameters</summary>

**Inputs**

| Name         | Type        | Description                                               |
|--------------|-------------|-----------------------------------------------------------|
| store_id     | String      | The unique identifier for the synthetic memory store. Required. |
| user_token   | SecretString| The authentication token for accessing the store. Required. |
| session_id   | String      | The chat session ID. Optional.                            |
| config       | Dictionary  | Additional configuration options for the memory store. Optional. |

**Outputs**

| Name             | Type                   | Description                                               |
|------------------|------------------------|-----------------------------------------------------------|
| memory_history   | SyntheticMemoryHistory | An instance of SyntheticMemoryHistory for the session.    |

</details>
"""
    )
    parser = AsyncLangflowDocsMarkdownParser()
    results = []
    async for result in parser._process_item(str(test_md)):
        results.append(result)

    assert len(results) > 0
    assert isinstance(results[0], dict)
    assert "SyntheticMemoryComponent" in results[0]


@pytest.mark.asyncio
async def test_async_python_component_parser(tmp_path):
    """Test AsyncPythonComponentParser parses a Python class with instance variables and public methods."""
    py_file = tmp_path / "component.py"
    py_file.write_text(
        '''
class TestComponent:
    """A test component for parsing."""
    
    inputs = {"param1": str, "param2": int}
    outputs = {"result": str}
    display_name = "Test Component"
    name = "test_component"
    
    def __init__(self):
        pass
        
    def process(self, param1: str, param2: int) -> str:
        """Process the inputs."""
        return f"{param1}_{param2}"
'''
    )

    parser = AsyncPythonComponentParser()
    results = []
    async for result in parser._process_item(str(py_file)):
        results.append(result)

    assert len(results) > 0
    assert isinstance(results[0], dict)
    # Check if we got the class information
    found_component = False
    for result in results:
        for key, value in result.items():
            if "TestComponent" in key:
                found_component = True
                assert "class_name" in value
                assert value["class_name"] == "TestComponent"
                break
    assert found_component


@pytest.mark.asyncio
async def test_async_python_sample_parser(tmp_path):
    """Test AsyncPythonSampleParser extracts function samples."""
    py_file = tmp_path / "samples.py"
    py_file.write_text(
        '''
import os
from pathlib import Path

def public_function(x: int) -> int:
    """A public function for testing."""
    return x * 2

def _private_function(x: int) -> int:
    """A private function that should be ignored."""
    return x * 3

async def async_function(name: str) -> str:
    """An async function for testing."""
    return f"Hello, {name}!"
'''
    )

    parser = AsyncPythonSampleParser()
    results = []
    async for result in parser._process_item(str(py_file)):
        results.append(result)

    assert len(results) >= 2  # Should have public_function and async_function

    # Check that we got the expected functions
    function_names = []
    for result in results:
        for key, value in result.items():
            function_names.append(value["name"])

    assert "public_function" in function_names
    assert "async_function" in function_names
    assert "_private_function" not in function_names  # Should be excluded
