import os
import ast
import pytest
from pathlib import Path
from dotenv import load_dotenv

from graphrag_agent.tools.content_parser import (
    AsyncLangflowDocsMarkdownParser,
    AsyncPythonComponentParser,
    AsyncNoParser,
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
    results = await parser._process_item(str(test_md))

    logger.debug(results)
    assert len(results) > 0
    assert isinstance(results, dict)
    assert "SyntheticMemoryComponent" in results.keys()


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
    results = await parser._process_item(str(py_file))

    assert len(results) > 0
    assert isinstance(results, dict)

    # Check if we got the class information
    assert any("TestComponent" in key for key in results.keys())

    # Get the component data (adjust key based on your actual parser output structure)
    component_key = "component.py::TestComponent"  # Fixed typo from "TestCompoment"
    component_data = results[component_key]

    # Assert each field
    assert component_data["class_name"] == "TestComponent"
    assert component_data["docstring"] == "A test component for parsing."
    assert component_data["inputs"] == "{'param1': str, 'param2': int}"
    assert component_data["outputs"] == "{'result': str}"
    assert component_data["display_name"] == "Test Component"
    assert component_data["name"] == "test_component"


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

    parser = AsyncNoParser()
    results = []
    results = await parser._process_item(str(py_file))
    import pdb

    pdb.set_trace()
    try:
        ast.parse(results["content"])
        assert True

    except SyntaxError:
        assert False
