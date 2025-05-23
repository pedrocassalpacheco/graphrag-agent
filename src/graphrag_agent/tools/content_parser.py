import ast
import asyncio
import httpx
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, override

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


class AsyncPageContentParser(BaseAsyncProcessor):
    """Asynchronous parser for web page content."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = httpx.AsyncClient()

    async def _get_content(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; GraphRagParser/1.0)"}
        response = await self.client.get(url, headers=headers, timeout=10.0)
        response.raise_for_status()
        return response.text

    async def _parse_html(self, html: str) -> Dict[str, List[str]]:
        pass

    @override
    async def _cleanup(self):
        await self.client.close()

    """ Specific handling for each HTML element that interest us. """

    def _clean_text(self, text: str) -> str:
        """
        Cleans unwanted control characters and other unnecessary symbols from text.
        """
        if not text:
            return ""
        # Clean control characters and normalize whitespace
        cleaned = re.sub(r"[\u0000-\u001F\u007F-\u009F\u00b6]", "", text)
        return re.sub(r"\s+", " ", cleaned).strip()

    def _handle_heading(self, element, content, current_title):
        """Handle heading tags (h1-h6). Returns new current_title."""
        current_title = self._clean_text(element.get_text(strip=True))
        content[current_title] = []

    def _handle_paragraph(self, element, content, current_title):
        """Handle paragraph tags."""
        if current_title:
            content[current_title].append(
                self._clean_text(element.get_text(strip=True))
            )

    def _handle_list(self, element, content, current_title):
        """Handle ul/ol tags."""
        if current_title:
            list_items = [
                self.clean_text(li.get_text(strip=True))
                for li in element.find_all("li")
            ]
            content[current_title].extend(list_items)

    def _handle_table(self, element, content, current_title):
        """Handle table tags."""
        if current_title:
            rows = []
            for row in element.find_all("tr"):
                cells = [
                    self._clean_text(cell.get_text(strip=True))
                    for cell in row.find_all(["td", "th"])
                ]
                rows.append(cells)
            content[current_title].append({"table": rows})

    def _handle_blockquote(self, element, content, current_title):
        """Handle blockquote tags."""
        if current_title:
            content[current_title].append(
                self._clean_text(element.get_text(strip=True))
            )

    def _handle_code(self, element, content, current_title):
        """Handle pre/code tags."""
        if current_title:
            content[current_title].append(self._clean_tex(element.get_text(strip=True)))

    def _handle_image(self, element, content, current_title):
        """Handle img tags."""
        if current_title:
            alt_text = self._clean_text(element.get("alt", "No description"))
            src = self._clean_text(element.get("src", "No source"))
            content[current_title].append({"image": {"alt": alt_text, "src": src}})

    def _get_handler(self, tag: str):
        """
        Returns the appropriate handler method for the given HTML tag.
        """
        return {
            "h1": self._handle_heading,
            "h2": self._handle_heading,
            "h3": self._handle_heading,
            "h4": self._handle_heading,
            "h5": self._handle_heading,
            "h6": self._handle_heading,
            "p": self._handle_paragraph,
            "ul": self._handle_list,
            "ol": self._handle_list,
            "table": self._handle_table,
            "blockquote": self._handle_blockquote,
            "pre": self._handle_code,
            "code": self._handle_code,
            "img": self._handle_image,
        }.get(tag)

    async def _parse_html(self, html: str) -> Dict[str, List[str]]:
        """Parse HTML content and extract structured data."""
        soup = BeautifulSoup(html, "html.parser")
        content = {}
        current_title = None
        for element in soup.find_all(HTML_TAGS_TO_PARSE):
            handler = self._get_handler(element.name)
            if handler:
                result = handler(element, content, current_title)
                if element.name.startswith("h"):
                    current_title = result
        return content

    @override
    async def _process_item(self, url: str) -> Dict[str, List[str]]:
        try:
            html = await self._get_content(url)
            return await self._parse_html(html)
        except httpx.HTTPError as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {str(e)}")
            return {}


class AsyncMarkdownParser(BaseAsyncProcessor):
    """
    Asynchronous parser for local markdown files.
    This class provides functionality to read and parse markdown files asynchronously.
    It extracts structured content such as headings, paragraphs, images, and tables
    directly from markdown content without converting it to HTML.
    Attributes:
        delay (float): The delay time (in seconds) for asynchronous operations.
    Methods:
        read_file(file_path: str) -> Optional[str]:
            Reads the content of a markdown file asynchronously.
        _get_content(file_path: str) -> Optional[str]:
            Reads content from a local file.
        _parse_content(content: str) -> Dict[str, List[str]]:
            Parses markdown content and extracts structured data such as headings,
            paragraphs, images, and tables.
    """

    @override
    async def _process_item(self, item: Any) -> Dict[str, List[str]]:
        if not isinstance(item, str):
            raise TypeError(f"Expected string URL, got {type(item).__name__}")
        else:
            file_path = str(item)

        """Read content from a local file."""
        try:
            content = await asyncio.to_thread(
                Path(file_path).read_text, encoding="utf-8"
            )
            results = await self._parse_markdown(content)
            return results
        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            logger.error(f"{type(e).__name__} when reading file {file_path}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error reading file {file_path}: {str(e)}")
            return {}

    async def _parse_markdown(self, content: str) -> Dict[str, List[str]]:
        """Parse markdown content directly without converting to HTML."""
        if not content:
            return {}

        structured_content = {}
        md = MarkdownIt("gfm-like")
        tokens = md.parse(content)

        structured_content = {}
        current_title = None
        current_content = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                # Save previous section
                if current_title:
                    structured_content[current_title] = "\n".join(
                        current_content
                    ).strip()
                    current_content = []

                # Get the heading content
                i += 1
                if i < len(tokens) and tokens[i].type == "inline":
                    current_title = tokens[i].content
                i += 1  # Skip heading_close
                continue

            elif token.type == "paragraph_open":
                i += 1
                if i < len(tokens) and tokens[i].type == "inline":
                    current_content.append(tokens[i].content)
                i += 1  # Skip paragraph_close

            elif token.type == "inline" and token.children:
                # Look for inline image tokens
                for child in token.children:
                    if child.type == "image":
                        alt = child.attrs.get("alt", "")
                        src = child.attrs.get("src", "")
                        img_text = f"![{alt}]({src})"
                        current_content.append(img_text)
                i += 1

            elif token.type == "image":
                # Standalone image token
                alt = token.attrs.get("alt", "")
                src = token.attrs.get("src", "")
                img_text = f"![{alt}]({src})"
                current_content.append(img_text)
                i += 1

            elif token.type == "table_open":
                table_lines = []
                while token.type != "table_close":
                    if token.type == "inline":
                        table_lines.append(token.content)
                    i += 1
                    if i < len(tokens):
                        token = tokens[i]
                    else:
                        break
                i += 1  # Skip table_close
                table_str = "\n".join(table_lines)
                current_content.append(table_str)
                continue

            else:
                i += 1

        if current_title and current_content:
            structured_content[current_title] = "\n".join(current_content).strip()

        logger.debug(f"Structured content: {structured_content}")
        logger.debug(f"Parsed {len(structured_content)} sections from markdown content")

        return structured_content


class AsyncLangflowDocsMarkdownParser(BaseAsyncProcessor):
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

    @override
    async def _process_item(self, item: Any) -> Dict[str, List[str]]:
        if not isinstance(item, str):
            raise TypeError(f"Expected string URL, got {type(item).__name__}")
        else:
            file_path = str(item)

        """Read content from a local file."""
        try:
            content = await asyncio.to_thread(
                Path(file_path).read_text, encoding="utf-8"
            )
            return await self._parse_markdown_component_doc(content)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None

    async def _parse_markdown_component_doc(self, content: str) -> Dict[str, str]:
        """Parse Markdown content, preserving tables and header hierarchy."""
        if not content:
            return {}

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

        # Save the last section
        if current_h2 and current_section:
            sections[current_h2] = current_section

        self.processed_count = len(sections)
        logger.info(f"Parsed {len(sections)} Markdown sections")

        return sections

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


class AsyncPythonComponentParser(BaseAsyncProcessor):
    """ """

    @override
    async def _process_item(self, item: Any) -> Any:
        if not isinstance(item, str):
            raise TypeError(f"Expected string URL, got {type(item).__name__}")
        else:
            file_path = str(item)

        path = Path(file_path)
        if not path.exists() or not path.is_file() or path.suffix != ".py":
            raise FileNotFoundError(f"File not found or invalid path: {file_path}")
        else:
            return await self._parse_python_component(path)

    async def _parse_python_component(self, path: Path) -> Dict[str, List[str]]:
        """Parse Python source code into a structured format using a file object."""
        structured_content = {}
        try:
            # Read the file content once
            with path.open("r", encoding="utf-8") as file:
                content = file.read()

            # Parse the file content into an AST
            tree = ast.parse(content, filename=str(path))

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

            self.processed_count += len(structured_content)
            logger.debug(
                f"Parsed {len(structured_content)} Python code sections from {path.name}"
            )
            logger.debug(f"Content \n{structured_content} ")

        except Exception as e:
            logger.error(f"Error parsing Python code from {path}: {e}")
            import traceback

            logger.debug(f"Call stack:\n{traceback.format_exc()}")

        return structured_content

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


class AsyncPythonSampleParser(BaseAsyncProcessor):
    """
    Asynchronous parser for Python code samples.

    Extracts complete function definitions with their docstrings as independent chunks.
    Each function is treated as a standalone sample without resolving dependencies.

    """

    async def _process_item(self, file: str) -> Dict[str, str]:
        try:
            # Read the file content
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            logger.debug(f"Processing file: {file}")
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
            structured_content = {}
            for node in ast.walk(tree):
                logger.debug(f"Analyzing {node}")
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and self.should_include_node(node):
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    function_code = "\n".join(lines[start_line:end_line])

                    # Create a unique key for this function
                    path = Path(file)
                    key = f"{path.name}::{node.name}"

                    # Get docstring
                    docstring = ast.get_docstring(node)

                    # Store function with imports and information about its source
                    structured_content[key] = {
                        "code": all_imports + "\n\n" + function_code,
                        "raw_code": function_code,  # Store original code without imports too
                        "imports": all_imports,
                        "docstring": docstring,
                        "name": node.name,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    }

            self.processed_count += len(structured_content)
            logger.debug(
                f"Extracted {len(structured_content)} function samples from {path.name}"
            )
            return structured_content

        except Exception as e:
            logger.error(f"Error parsing {path}: {e}")
            return {}

    def should_include_node(self, node: ast.AST) -> bool:
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


class PythonAPIExtractor(AsyncPythonComponentParser):
    """Extracts class API details without implementation code."""

    async def _parse_content(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """Parse Python file to extract only API-level details."""
        api_structure = {}

        try:
            # Read file once
            with path.open("r", encoding="utf-8") as file:
                content = file.read()

            # Parse AST
            tree = ast.parse(content, filename=str(path))

            # Extract classes and their methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    class_key = f"{path.name}::{class_name}"

                    # Basic class info
                    api_structure[class_key] = {
                        "name": class_name,
                        "docstring": ast.get_docstring(node),
                        "methods": {},
                        "constructor": None,
                        "base_classes": [ast.unparse(base) for base in node.bases],
                    }

                    # Extract methods
                    for method_node in node.body:
                        if isinstance(
                            method_node, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            method_info = self._extract_function_signature(method_node)

                            # Special handling for constructor
                            if method_node.name == "__init__":
                                api_structure[class_key]["constructor"] = method_info
                            else:
                                api_structure[class_key]["methods"][
                                    method_node.name
                                ] = method_info

            self.processed_count += len(api_structure)
            return api_structure

        except Exception as e:
            logger.error(f"Error extracting API from {path}: {e}")
            return {}
