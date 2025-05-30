import ast
import asyncio
import httpx
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, override, AsyncGenerator

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

from graphrag_agent.utils.logging_config import get_logger
from graphrag_agent.tools.async_processor import BaseAsyncProcessor

logger = get_logger(__name__)

HTML_TAGS_TO_PARSE = [
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "ul",
    "ol",
    "table",
    "blockquote",
    "pre",
    "code",
    "img",
]


class ContentFormatMixin:
    """Mixin to provide content formatting utilities.
    This done to ensure different sources are formatted in a consistent way.
    source: Source of the content
    title: Title of the content
    content: the actual document or document chunk
    """

    def format_content(self, source: str, content: Dict[str, List[str]]) -> str:
        for title, section_content in content.items():
            if not section_content:
                continue

            logger.debug(f"Parsed section: {title} from {source}")

            return {
                "source": source,  # Using item as the identifier
                "title": title,
                "content": section_content,
            }


class AsyncLangflowDocsMarkdownParser(BaseAsyncProcessor, ContentFormatMixin):
    """A parser for processing Markdown content specific for langflow
    asynchronously, specifically designed to handle Langflow documentation. This parser preserves tables and header
    hierarchy while extracting content into structured sections.
    Methods:
        async _parse_content(content: str) -> Dict[str, str]:
            Parses the provided Markdown content, organizing it into sections
            based on H2 headers. Each section includes its content, preserving
            tables and subheaders (H3).
        _process_table(tokens, i) -> Tuple[List[str], int]:
            Processes a Markdown table, converting it into a list of formatted
            Markdown rows. Handles table headers, alignment, and data rows.
    """

    async def _parse_markdown_component_doc(
        self, content: str
    ) -> AsyncGenerator[Any, None]:
        """Parse Markdown content, preserving tables and header hierarchy."""
        if not content or not isinstance(content, str):
            logger.error("Invalid content provided for parsing.")
            return

        md = MarkdownIt("gfm-like")
        tokens = md.parse(content)

        sections = {}
        current_h2 = None
        current_section = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check for H2 headers
            if token.type == "heading_open" and token.tag == "h2":
                # Save previous section if it exists
                if current_h2 and current_section:
                    sections[current_h2] = current_section
                    yield {
                        current_h2: current_section,
                    }
                    current_section = []

                # Get the H2 heading content
                i += 1
                if i < len(tokens) and tokens[i].type == "inline":
                    current_h2 = tokens[i].content.strip()
                i += 1  # Skip heading_close
                continue

            # Handle tables specially to preserve structure
            elif current_h2 and token.type == "table_open":
                table_md, new_i = self._process_table(tokens, i)
                current_section.append(table_md)
                i = new_i

            # For all other content in an H2 section
            elif current_h2:
                if token.type == "paragraph_open":
                    i += 1
                    if i < len(tokens) and tokens[i].type == "inline":
                        current_section.append(tokens[i].content)
                    i += 1  # Skip paragraph_close

                elif token.type == "heading_open" and token.tag == "h3":
                    # Include H3 headers as part of the content
                    i += 1
                    if i < len(tokens) and tokens[i].type == "inline":
                        current_section.append(f"### {tokens[i].content}")
                    i += 1  # Skip heading_close

                elif token.type == "inline" and token.content:
                    current_section.append(token.content)
                    i += 1

                else:
                    i += 1
            else:
                i += 1

        # Yield the last section
        if current_h2 and current_section:
            yield {
                current_h2: current_section,
            }

        self.processed_count = len(sections)
        logger.info(f"Parsed {len(sections)} Markdown sections")

    def _process_table(self, tokens, i):
        """
        Process a Markdown table and format it as a list of table rows.

        Args:
            tokens: List of markdown tokens
            i: Current token index

        Returns:
            tuple: (list of table row strings, new token index)
        """
        table_rows = []
        header_row = []
        alignment = []
        is_header = True

        i += 1  # Move to next token after table_open

        # Process table content
        while i < len(tokens) and tokens[i].type != "table_close":
            if tokens[i].type == "thead_open":
                i += 1  # Skip thead_open
            elif tokens[i].type == "thead_close":
                is_header = False
                i += 1  # Skip thead_close
            elif tokens[i].type == "tbody_open":
                i += 1  # Skip tbody_open
            elif tokens[i].type == "tbody_close":
                i += 1  # Skip tbody_close
            elif tokens[i].type == "tr_open":
                row = []
                i += 1  # Skip tr_open

                # Process row content
                while i < len(tokens) and tokens[i].type != "tr_close":
                    if tokens[i].type in ["th_open", "td_open"]:
                        if tokens[i].type == "th_open" and is_header:
                            # Extract alignment if this is a header cell
                            align = (
                                tokens[i]
                                .attrs.get("style", "")
                                .replace("text-align:", "")
                                .strip()
                                if tokens[i].attrs
                                else ""
                            )
                            alignment.append(align)

                        i += 1  # Skip th_open or td_open

                        # Get cell content
                        if i < len(tokens) and tokens[i].type == "inline":
                            cell_content = tokens[i].content.strip()
                            row.append(cell_content)

                        i += 1  # Skip cell content
                        i += 1  # Skip th_close or td_close
                    else:
                        i += 1  # Skip other tokens

                if is_header:
                    header_row = row
                else:
                    table_rows.append(row)

                i += 1  # Skip tr_close
            else:
                i += 1  # Skip other tokens

        # Format the table in Markdown as a list of rows
        table_md_rows = []
        if header_row:
            # Create the header row
            table_md_rows.append("| " + " | ".join(header_row) + " |")

            # Create the separator row with alignment
            sep_row = []
            for idx, col in enumerate(header_row):
                align = alignment[idx] if idx < len(alignment) else ""
                if align == "center":
                    sep_row.append(":" + "-" * (len(col) + 1) + ":")
                elif align == "right":
                    sep_row.append("-" * (len(col) + 1) + ":")
                else:  # left or default
                    sep_row.append("-" * (len(col) + 2))

            table_md_rows.append("| " + " | ".join(sep_row) + " |")

            # Add data rows
            for row in table_rows:
                # Make sure each row has the same number of columns as the header
                while len(row) < len(header_row):
                    row.append("")
                table_md_rows.append("| " + " | ".join(row) + " |")

        return (
            table_md_rows,
            i + 1,
        )  # Return the list of formatted table rows and the new index

    @override
    async def _process_item(self, item: Any) -> AsyncGenerator[Any, None]:
        """Process item and yield individual sections."""
        if not isinstance(item, str):
            raise TypeError(f"Expected string path, got {type(item).__name__}")
        else:
            file_path = str(item)

        try:
            content = await asyncio.to_thread(
                Path(file_path).read_text, encoding="utf-8"
            )

            # Use async for to iterate over the generator and yield each section
            async for section in self._parse_markdown_component_doc(content):
                yield section

        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            logger.error(f"{type(e).__name__} when reading file {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")


class AsyncPythonComponentParser(BaseAsyncProcessor, ContentFormatMixin):
    """ """

    async def _parse_python_component(self, path: Path) -> AsyncGenerator[Any, None]:
        """Parse Python source code into a structured format using a file object."""
        structured_content = {}
        try:
            # Read the file content once
            with path.open("r", encoding="utf-8") as file:
                content = file.read()

            # Parse the file content into an AST
            tree = ast.parse(content)

            # Extract module docstring if available
            module_docstring = ast.get_docstring(tree)
            if module_docstring:
                structured_content[f"{path.name}::module"] = [module_docstring]

            # First pass: process class definitions
            class_data = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._extract_class_definition(node)
                    class_key = f"{path.name}::{class_info['class_name']}"

                    # Initialize with class data and empty public_methods list
                    class_data[class_key] = class_info

                    # Store in structured content
                    structured_content[class_key] = class_data[class_key]
                    yield {class_key: class_data[class_key]}

            self.processed_count += len(structured_content)
            logger.debug(
                f"Parsed {len(structured_content)} Python code sections from {path.name}"
            )
            logger.debug(f"Content \n{structured_content} ")

        except Exception as e:
            logger.error(f"Error parsing Python code from {path}: e")
            import traceback

            logger.debug(f"Call stack:\n{traceback.format_exc()}")

    def _extract_function_signature(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> Dict[str, Any]:
        """
        Extract the function signature and related information from a function node.

        Args:
            node (Union[ast.FunctionDef, ast.AsyncFunctionDef]): The AST node representing a function.

        Returns:
            Dict[str, Any]: Dictionary containing function signature and metadata.
        """
        # Get the function name and arguments
        func_name = node.name
        args = ast.unparse(node.args)

        # Get the return type if available
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"

        # Construct the signature
        signature = f"{func_name}{args}{return_type}"

        # Extract docstring if available
        docstring = ast.get_docstring(node)

        # Extract decorators if available
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(ast.unparse(decorator))

        # Return a dictionary with the signature as key and additional information
        return {
            "method_signature": signature,
            "name": func_name,
            "docstring": docstring,
            "decorators": decorators,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }

    def _extract_class_definition(self, node: ast.ClassDef) -> Dict[str, Any]:
        """
        Handle class definitions and extract relevant information.
        Args:
            node (ast.ClassDef): The AST node representing a class definition.
        Returns:
            List[str]: A list of strings representing the class definition.
        """
        class_name = node.name
        class_vars = {"inputs": {}, "outputs": {}, "display_name": None, "name": None}

        # Extract base classes
        base_classes = []
        for base in node.bases:
            base_class_name = ast.unparse(base)
            base_classes.append(base_class_name)

        # Extract class-level variables
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if target.id in {"inputs", "outputs"}:
                            class_vars[target.id] = ast.unparse(stmt.value)
                        elif target.id == "display_name":
                            class_vars["display_name"] = ast.literal_eval(stmt.value)
                        elif target.id == "name":
                            class_vars["name"] = ast.literal_eval(stmt.value)

        # Add to structured content
        return {
            "class_name": class_name,
            "docstring": ast.get_docstring(node),
            "inputs": class_vars["inputs"],
            "outputs": class_vars["outputs"],
            "display_name": class_vars["display_name"],
            "name": class_vars["name"],
        }

    def _should_include_node(self, node: ast.AST) -> bool:
        """
        Determine if an AST node should be included based on its name.

        Args:
            node (ast.AST): The AST node to check.

        Returns:
            bool: True if the node should be included, False otherwise.
        """
        # Check if the node has a name attribute and if it starts with an underscore
        return hasattr(node, "name") and not node.name.startswith("_")

    @override
    async def _process_item(self, item: Any) -> AsyncGenerator[Any, None]:

        if not isinstance(item, str):
            raise TypeError(f"Expected string URL, got {type(item).__name__}")
        else:
            file_path = str(item)

        try:
            path = Path(file_path)
            # Use async for to iterate over the generator and yield each section
            async for section in self._parse_python_component(path):
                yield section

        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            logger.error(f"{type(e).__name__} when reading file {file_path}: {e}")

        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")


# Fix AsyncPythonSampleParser - it returns instead of yields
class AsyncPythonSampleParser(BaseAsyncProcessor, ContentFormatMixin):
    """
    Asynchronous parser for Python code samples.

    Extracts complete function definitions with their docstrings as independent chunks.
    Each function is treated as a standalone sample without resolving dependencies.
    """

    @override
    async def _process_item(self, item: Any) -> AsyncGenerator[Any, None]:
        if not isinstance(item, str):
            raise TypeError(f"Expected string path, got {type(item).__name__}")

        file_path = str(item)

        try:
            # Read the file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Processing file: {file_path}")
            logger.debug(f"File content: {content}")

            # Parse the AST
            tree = ast.parse(content)
            lines = content.splitlines()

            # First, collect all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    start_line = node.lineno - 1
                    end_line = getattr(node, "end_lineno", start_line + 1)
                    import_code = "\n".join(lines[start_line:end_line])
                    imports.append(import_code)

            # Join all imports
            all_imports = "\n".join(imports)
            logger.debug(f"Imports found: {all_imports}")

            # Now process functions with imports included
            for node in ast.walk(tree):
                logger.debug(f"Analyzing {node}")
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and self._should_include_node(node):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    function_code = "\n".join(lines[start_line:end_line])

                    # Create a unique key for this function
                    path = Path(file_path)
                    key = f"{path.name}::{node.name}"

                    # Get docstring
                    docstring = ast.get_docstring(node)

                    # Yield each function as a separate item
                    yield {
                        key: {
                            "code": all_imports + "\n\n" + function_code,
                            "raw_code": function_code,
                            "imports": all_imports,
                            "docstring": docstring,
                            "name": node.name,
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                        }
                    }

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

    def _should_include_node(self, node: ast.AST) -> bool:
        """
        Determine if a node should be included in the extraction.

        Excludes private methods (starting with underscore) and special methods.

        Args:
            node (ast.AST): The AST node to check.

        Returns:
            bool: True if the node should be included, False otherwise.
        """
        if not hasattr(node, "name"):
            return False

        # Skip private methods and special methods
        if node.name.startswith("_"):
            return False

        # Only include function definitions
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False

        return True
