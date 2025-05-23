import asyncio
import os
import json
from pathlib import Path
import pytest
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
    result = await parser._process_item(str(test_md))

    assert isinstance(result, dict)
    assert "Title" in result or "Subtitle" in result


@pytest.mark.asyncio
async def test_async_markdown_parser_on_nonexistent_file(tmp_path):
    """Test AsyncMarkdownParser returns empty dict for nonexistent file."""
    parser = AsyncMarkdownParser()
    result = await parser._process_item(str(tmp_path / "does_not_exist.md"))
    assert isinstance(result, dict)
    assert len(result) == 0


@pytest.mark.asyncio
async def test_async_page_content_parser_on_html(httpserver):
    """Test AsyncPageContentParser parses HTML from a mocked HTTP server."""
    html_content = "<h1>Header</h1><p>Paragraph</p>"
    httpserver.expect_request("/test.html").respond_with_data(
        html_content, content_type="text/html"
    )

    url = httpserver.url_for("/test.html")
    parser = AsyncPageContentParser()
    result = await parser._process_item(url)
    logger.debug(result)
    assert isinstance(result, dict)
    assert any("Header" in k or "Paragraph" in v for k, v in result.items())


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
    result = await parser._process_item(str(test_md))

    assert isinstance(result, dict)
    assert "SyntheticMemoryComponent" in result


@pytest.mark.asyncio
async def test_async_python_component_parser(tmp_path):
    """Test AsyncPythonComponentParser parses a Python class with instance variables and public methods."""
    py_file = tmp_path / "component.py"
    py_file.write_text(
        """
class ApifyActorsComponent:
    display_name = "Apify Actors"
    description = "Use Apify Actors to extract data from hundreds of places fast."
    name = "ApifyActors"

    inputs = [
        SecretStrInput(
            name="apify_token",
            display_name="Apify Token",
        ),
        StrInput(
            name="actor_id",
            display_name="Actor",
        ),
    ]

    outputs = [
        Output(display_name="Output", name="output"),
        Output(display_name="Tool", name="tool"),
    ]

    def public_method(self):
        return "public"

    def another_public_method(self, x):
        return x

    def _private_method(self):
        return "private"
"""
    )

    parser = AsyncPythonComponentParser()
    result = await parser._process_item(str(py_file))

    assert isinstance(result, dict)
    # Check for class name
    assert "component.py::ApifyActorsComponent" in result
    component = result["component.py::ApifyActorsComponent"]
    # Check for class name
    assert component["class_name"] == "ApifyActorsComponent"
    # Check for instance/class variables
    assert component["display_name"] == "Apify Actors"
    assert component["name"] == "ApifyActors"
    # Check for inputs and outputs as string representations
    assert "apify_token" in component["inputs"]
    assert "actor_id" in component["inputs"]
    assert "Output" in component["outputs"]
    assert "Tool" in component["outputs"]


@pytest.mark.asyncio
async def test_async_python_sample_parser(tmp_path):
    """Test AsyncPythonSampleParser parses a Python sample file with a document_qa_graph function."""
    py_file = tmp_path / "sample.py"
    py_file.write_text(
        """
from langflow.components.data import FileComponent
from langflow.components.inputs import ChatInput
from langflow.components.models import OpenAIModelComponent
from langflow.components.outputs import ChatOutput
from langflow.components.processing import ParseDataComponent
from langflow.components.prompts import PromptComponent
from langflow.graph import Graph

def document_qa_graph(template: str | None = None):
    if template is None:
        template = '''Answer user's questions based on the document below:

---

{Document}

---

Question:
{Question}

Answer:
'''
    file_component = FileComponent()
    parse_data_component = ParseDataComponent()
    parse_data_component.set(data=file_component.load_files)

    chat_input = ChatInput()
    prompt_component = PromptComponent()
    prompt_component.set(
        template=template,
        context=parse_data_component.parse_data,
        question=chat_input.message_response,
    )

    openai_component = OpenAIModelComponent()
    openai_component.set(input_value=prompt_component.build_prompt)

    chat_output = ChatOutput()
    chat_output.set(input_value=openai_component.text_response)

    return Graph(start=chat_input, end=chat_output)
"""
    )

    parser = AsyncPythonSampleParser()
    result = await parser._process_item(str(py_file))

    assert isinstance(result, dict)
    # Check for the function name in the result
    assert any("document_qa_graph" in k for k in result.keys())
    # Optionally, check that the code contains key components
    assert any("FileComponent" in v["code"] for v in result.values())
    assert any("Graph" in v["code"] for v in result.values())
