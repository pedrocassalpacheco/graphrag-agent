import ast
from pathlib import Path
from typing import Any, Dict, Union, override

from markdown_it import MarkdownIt

from graphrag_agent.utils.logging_config import get_logger
from graphrag_agent.tools.async_processor import BaseAsyncProcessor
from graphrag_agent.tools.parse_content import ParsedContent

logger = get_logger(__name__)


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

    async def _parse_markdown_component_doc(
        self, file_path: str
    ) -> list[ParsedContent]:
        """Parse Markdown content, preserving tables and header hierarchy."""
        if not file_path or not isinstance(file_path, str):
            logger.error("Invalid content provided for parsing.")
            return []

        content = await self._read_file_content(file_path)

        md = MarkdownIt("gfm-like")
        tokens = md.parse(content)

        parsed_content = []
        current_h2 = None
        current_section = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check for H2 headers
            if token.type == "heading_open" and token.tag == "h2":
                # Save previous section if it exists
                if current_h2 and current_section:
                    parsed_content.append(
                        ParsedContent(
                            source=file_path,
                            section=current_h2,
                            content="\n".join(current_section),
                            metadata={
                                "parser": "langflow_docs_markdown",
                                "level": "h2",
                            },
                            content_type="heading",
                        )
                    )

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
                current_section.extend(table_md)
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
            parsed_content.append(
                ParsedContent(
                    source=file_path,
                    section=current_h2,
                    content="\n".join(current_section),
                    metadata={"parser": "langflow_docs_markdown", "level": "h2"},
                    content_type="heading",
                )
            )

        logger.info(f"Parsed {len(parsed_content)} Markdown sections")
        return parsed_content

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
    async def _process_item(self, item: Any) -> list[ParsedContent]:
        """Process item and yield individual sections."""
        if not isinstance(item, str):
            raise TypeError(f"Expected string path, got {type(item).__name__}")
        else:
            file_path = str(item)

        try:
            # Use async for to iterate over the generator and yield each section
            return await self._parse_markdown_component_doc(file_path)

        except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
            # Handle file-related errors specifically
            logger.error(f"File error parsing {file_path}: {e}")
            return []  # Return empty list for file errors

        except Exception as e:
            # Handle parsing errors differently - maybe you want to fail fast here
            logger.error(f"Parse error in {file_path}: {e}")
            raise  # Re-raise parsing errors


class AsyncPythonComponentParser(BaseAsyncProcessor):
    """Parser for Python component files that extracts class definitions."""

    async def _parse_python_component(self, file_path: str) -> list[ParsedContent]:
        """Parse Python source code from file path into ParsedContent objects."""
        content = await self._read_file_content(file_path)
        tree = ast.parse(content)
        parsed_sections = []

        # Extract all imports once
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))

        # Process each class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node, imports)

                parsed_sections.append(
                    ParsedContent(
                        source=file_path,
                        section=f"class::{class_info['class_name']}",
                        content=class_info["formatted_content"],
                        metadata={
                            "parser": "python_component",
                            **class_info["metadata"],
                        },
                        content_type="class",
                    )
                )

        logger.info(f"Parsed {len(parsed_sections)} Python sections from {file_path}")
        return parsed_sections

    def _extract_class_info(self, node: ast.ClassDef, imports: list) -> dict:
        """Extract all class information in a clean, organized way."""
        class_name = node.name
        docstring = ast.get_docstring(node) or ""
        base_classes = [ast.unparse(base) for base in node.bases]

        # Define what attributes we're looking for
        attributes = {
            "inputs": "",
            "outputs": "",
            "display_name": "",
            "name": "",
            "description": "",
            "icon": "",
        }

        public_methods = []

        # Process all statements in the class body
        for stmt in node.body:
            # Handle function definitions
            if isinstance(stmt, ast.FunctionDef) and not stmt.name.startswith("_"):
                args = [arg.arg for arg in stmt.args.args]
                signature = f"def {stmt.name}({', '.join(args)})"
                public_methods.append(signature)
                continue

            # Extract target and value from assignments
            target_value_pairs = self._get_assignment_pairs(stmt)

            for target_name, value in target_value_pairs:
                if target_name in attributes:
                    if target_name in ["inputs", "outputs"]:
                        attributes[target_name] = ast.unparse(value)
                    else:
                        # For other attributes, try to get the literal value
                        try:
                            attributes[target_name] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            attributes[target_name] = ast.unparse(value)

        # Format the content
        formatted_content = self._format_class_as_code(
            class_name=class_name,
            docstring=docstring,
            base_classes=base_classes,
            public_methods=public_methods,
            imports=imports,
            **attributes,
        )

        return {
            "class_name": class_name,
            "formatted_content": formatted_content,
            "metadata": {
                "class_name": class_name,
                "base_classes": base_classes,
                "public_methods": public_methods,
                "imports": imports,
                **attributes,
            },
        }

    def _get_assignment_pairs(self, stmt: ast.stmt) -> list[tuple[str, ast.expr]]:
        """Extract (target_name, value) pairs from assignment statements."""
        pairs = []

        if isinstance(stmt, ast.Assign):
            # Regular assignment: name = "value"
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    pairs.append((target.id, stmt.value))

        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            # Annotated assignment: name: str = "value"
            if stmt.value is not None:  # Skip annotation-only assignments
                pairs.append((stmt.target.id, stmt.value))

        return pairs

    def _format_class_as_code(
        self,
        class_name: str,
        docstring: str,
        base_classes: list,
        inputs: str,
        outputs: str,
        display_name: str,
        name: str,
        description: str,
        icon: str,
        public_methods: list,
        imports: list,
    ) -> str:
        """Format class information to look like actual Python code."""

        lines = []

        # Add imports at the top
        if imports:
            lines.extend(imports)
            lines.append("")  # Empty line after imports

        # Add class definition with base classes
        if base_classes:
            base_classes_str = f"({', '.join(base_classes)})"
        else:
            base_classes_str = ""

        lines.append(f"class {class_name}{base_classes_str}:")

        # Add class docstring
        if docstring:
            lines.append('    """')
            docstring_lines = docstring.split("\n")
            for doc_line in docstring_lines:
                lines.append(f"    {doc_line}")
            lines.append('    """')
            lines.append("")

        # Add class attributes
        if display_name:
            lines.append(f'    display_name: str = "{display_name}"')
        if description:
            lines.append(f'    description: str = "{description}"')
        if icon:
            lines.append(f'    icon = "{icon}"')
        if name:
            lines.append(f'    name = "{name}"')

        lines.append("")

        # Format inputs with proper line breaks
        if inputs and inputs != "{}":
            lines.append("    inputs = [")
            # Split by comma and format each input on its own line
            input_items = inputs.strip("[]").split(", ")
            for i, item in enumerate(input_items):
                if item.strip():
                    comma = "," if i < len(input_items) - 1 else ""
                    lines.append(f"        {item.strip()}{comma}")
            lines.append("    ]")
            lines.append("")

        # Format outputs with proper line breaks
        if outputs and outputs != "{}":
            lines.append("    outputs = [")
            # Split by comma and format each output on its own line
            output_items = outputs.strip("[]").split(", ")
            for i, item in enumerate(output_items):
                if item.strip():
                    comma = "," if i < len(output_items) - 1 else ""
                    lines.append(f"        {item.strip()}{comma}")
            lines.append("    ]")
            lines.append("")

        # Add public method signatures
        if public_methods:
            for method in public_methods:
                lines.append(f"    {method}:")
                lines.append('        """Method implementation..."""')
                lines.append("        pass")
                lines.append("")

        return "\n".join(lines)

    @override
    async def _process_item(self, item: Any) -> list[ParsedContent]:
        """Process item with comprehensive error handling."""
        if not isinstance(item, str):
            raise TypeError(f"Expected string path, got {type(item).__name__}")

        file_path = str(item)

        try:
            return await self._parse_python_component(file_path)

        except FileNotFoundError:
            logger.warning(f"File not found, skipping: {file_path}")
            return []
        except PermissionError:
            logger.warning(f"Permission denied, skipping: {file_path}")
            return []
        except UnicodeDecodeError:
            logger.warning(f"Encoding error, skipping: {file_path}")
            return []
        except SyntaxError as e:
            logger.warning(f"Python syntax error in {file_path}, skipping: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            return []


class AsyncNoParser(BaseAsyncProcessor):
    """Simply return the file content without any processing."""

    @override
    async def _process_item(self, item: Any) -> list[ParsedContent]:
        """Process item with comprehensive error handling."""
        if not isinstance(item, str):
            raise TypeError(f"Expected string path, got {type(item).__name__}")

        file_path = str(item)

        try:
            content = await self._read_file_content(file_path)

            return [
                ParsedContent(
                    source=file_path,
                    section="file",
                    content=content,
                    metadata={"parser": "no_parser"},
                    content_type="file",
                )
            ]

        except FileNotFoundError:
            logger.warning(f"File not found, skipping: {file_path}")
            return []
        except PermissionError:
            logger.warning(f"Permission denied, skipping: {file_path}")
            return []
        except UnicodeDecodeError:
            logger.warning(f"Encoding error, skipping: {file_path}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {e}")
            return []


class AsyncPythonExampleParser(BaseAsyncProcessor):
    """
    Asynchronous parser for Python code samples.

    Extracts complete function definitions with their docstrings as independent chunks.
    Each function is treated as a standalone sample without resolving dependencies.
    """

    @override
    async def _process_item(self, item: Any) -> Any:
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
