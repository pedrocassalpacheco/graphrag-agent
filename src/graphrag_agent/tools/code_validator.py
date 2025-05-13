import asyncio
import ast
import re
from pathlib import Path
from typing import Dict, List, Union, Any, Optional
import traceback

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseAsyncCodeValidator:
    """Base class for asynchronous code validators."""

    def __init__(self):
        self.validated_count = 0

    def _extract_code_from_markdown(self, text: str) -> tuple:
        """Extract code from markdown code blocks."""
        pattern = r"```(?:python)?\s*([\s\S]*?)```"
        code_blocks = re.findall(pattern, text)

        if not code_blocks:
            return text, False

        extracted_code = "\n\n".join(code_blocks)
        return extracted_code, True

    async def validate(
        self,
        input: Union[asyncio.Queue, str],
        output: Optional[Union[asyncio.Queue, str, Path]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process input and validate code, with optional output.

        Args:
            input: Either a queue containing code to validate or a code string
            output: Optional queue or file path for results

        Returns:
            Dict with validation results if input is a string and no output specified
        """
        # Handle direct string validation
        if isinstance(input, str):
            result = await self._validate_code(input)

            # Handle different output types
            if isinstance(output, (str, Path)) and result["is_valid"]:
                await self._write_to_file(result, Path(output))
            elif isinstance(output, asyncio.Queue):
                await output.put(result)
                return None

            return result

        # Queue-based processing
        while True:
            try:
                code = await input.get()
                if code is None:  # None signals end of processing
                    if isinstance(output, asyncio.Queue):
                        await output.put(None)  # Signal downstream processors
                    break

                result = await self._validate_code(code)

                # Handle different output types
                if isinstance(output, (str, Path)) and result["is_valid"]:
                    await self._write_to_file(result, Path(output))
                elif isinstance(output, asyncio.Queue):
                    await output.put(result)

                input.task_done()

            except Exception as e:
                logger.error(f"Error in validation: {str(e)}")
                logger.error(traceback.format_exc())
                error_result = {"error": str(e), "is_valid": False}

                if isinstance(output, asyncio.Queue):
                    await output.put(error_result)

        return None

    async def _write_to_file(self, result: Dict[str, Any], file_path: Path):
        """Write validated code to a file."""
        # Get the code to write
        code_to_write = result.get("extracted_code", result.get("original_code", ""))

        if not code_to_write:
            logger.warning("No code to write to file")
            return

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Choose mode based on file existence
        mode = "a" if file_path.exists() else "w"
        separator = (
            "\n\n# " + "=" * 50 + "\n# NEW CODE BLOCK\n\n" if mode == "a" else ""
        )

        try:
            with open(file_path, mode) as f:
                f.write(f"{separator}{code_to_write}")

            logger.info(
                f"{'Appended to' if mode == 'a' else 'Created'} file: {file_path}"
            )
            result["file_path"] = str(file_path)

        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {e}")
            result["file_error"] = str(e)

    async def _validate_code(self, code: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main validation method to be implemented by subclasses.

        Args:
            code: Code to validate

        Returns:
            Validation result dictionary
        """
        raise NotImplementedError("Subclasses must implement this method")


class PythonCodeValidator(BaseAsyncCodeValidator):
    """Python-specific code validator."""

    def __init__(self, check_imports: bool = True, strict_mode: bool = False):
        super().__init__()
        self.check_imports = check_imports
        self.strict_mode = strict_mode

    async def _validate_code(self, code: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Validate Python code."""
        # Extract code string if input is a dictionary
        if isinstance(code, dict):
            code_str = code.get("code", "")
            if not code_str:
                return {
                    "is_valid": False,
                    "errors": ["No code provided"],
                    "warnings": [],
                    "original_code": code_str,
                }
        else:
            code_str = code

        # Extract code if it's markdown
        original_code = code_str
        extracted_code, is_markdown = self._extract_code_from_markdown(code_str)

        # Initialize result
        result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "original_code": original_code,
        }

        if is_markdown:
            result["extracted_code"] = extracted_code

        # Code to validate
        code_to_validate = extracted_code if is_markdown else original_code

        # Empty code check
        if not code_to_validate.strip():
            result["is_valid"] = False
            result["errors"].append("No code content found")
            return result

        # Check syntax using Python's built-in compile
        try:
            compile(code_to_validate, "<string>", "exec")
        except SyntaxError as e:
            result["is_valid"] = False
            result["errors"].append(
                f"Syntax error at line {e.lineno}, position {e.offset}: {e.msg}"
            )
            return result

        # Perform additional validation
        try:
            tree = ast.parse(code_to_validate)

            # Add strict checks if requested
            if self.strict_mode:
                warnings = await self._check_strict_issues(tree)
                result["warnings"].extend(warnings)

            # Validate imports if requested
            if self.check_imports:
                import_warnings = await self._validate_imports(code_to_validate)
                result["warnings"].extend(import_warnings)

            # Increment validated count
            self.validated_count += 1

        except Exception as e:
            logger.error(f"Error during AST validation: {e}")
            result["warnings"].append(f"Error during code analysis: {str(e)}")

        return result

    async def _check_strict_issues(self, tree: ast.AST) -> List[str]:
        """Check for code issues that might indicate problems."""
        warnings = []

        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                warnings.append(
                    "Bare 'except:' clause - consider catching specific exceptions"
                )

        return warnings

    async def _validate_imports(self, code: str) -> List[str]:
        """Validate import statements without executing the full code."""
        import_warnings = []

        # Extract only import statements
        import_lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and not stripped.startswith(
                "#"
            ):
                import_lines.append(line)

        if not import_lines:
            return []

        # Create a new code string with just the imports
        import_code = "\n".join(import_lines)

        # Try to execute just the imports
        try:
            # Use a namespace to avoid polluting our environment
            namespace = {}
            exec(import_code, namespace)
        except Exception as e:
            import_warnings.append(f"Import issue: {str(e)}")

        return import_warnings


async def validate_code(
    code: str,
    check_imports: bool = True,
    strict_mode: bool = False,
    output_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to validate code.

    Args:
        code: Code to validate (can be markdown or raw code)
        check_imports: Whether to validate import statements
        strict_mode: Whether to perform stricter validation checks
        output_file: Optional file path to write validated code

    Returns:
        Validation results dictionary
    """
    validator = PythonCodeValidator(
        check_imports=check_imports, strict_mode=strict_mode
    )
    return await validator.validate(code, output_file)
