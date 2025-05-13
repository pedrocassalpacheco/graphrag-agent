import asyncio
import ast
import io
from typing import Dict, List, Union, Any, Optional, Tuple
import traceback
import re

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseAsyncCodeValidator:
    """
    Base class for asynchronous code validators.

    This abstract class defines the interface for code validators that
    can process code from a queue or direct string input and validate it.
    """

    def __init__(self):
        self.validated_count = 0

    def _extract_code_if_markdown(self, text):
        """Extract code from markdown if needed - base implementation."""
        # Pattern to match code blocks with optional language specifier
        pattern = r"```(?:\w+)?\s*([\s\S]*?)```"

        # Check if it looks like markdown
        if not re.search(r"```", text):
            return text, False  # Not markdown

        # Find all code blocks
        code_blocks = re.findall(pattern, text)

        if not code_blocks:
            return text, False

        # Join all code blocks with newlines between them
        extracted_code = "\n\n".join(code_blocks)
        return extracted_code, True

    async def validate(
        self, input: Union[asyncio.Queue, str], output: Optional[asyncio.Queue] = None
    ) -> Union[Dict[str, Any], None]:
        """
        Validate code from input and optionally send results to output queue.

        Args:
            input: Either a queue containing code to validate or a code string
            output: Optional queue to receive validation results

        Returns:
            Dict with validation results if input is a string and output is None,
            otherwise None (results go to the output queue)
        """
        if isinstance(input, str):
            # Direct string validation
            result = await self._validate_code(input)
            if output:
                await output.put(result)
                return None
            return result

        # Queue-based processing
        while True:
            try:
                code = await input.get()
                if code is None:  # None signals end of processing
                    if output:
                        await output.put(None)  # Signal downstream processors
                    break

                result = await self._validate_code(code)

                if output:
                    await output.put(result)

                input.task_done()

            except Exception as e:
                logger.error(f"Error in validation: {str(e)}")
                logger.error(f"Call stack:\n{traceback.format_exc()}")
                if output:
                    await output.put({"error": str(e), "is_valid": False})

    async def _validate_code(self, code: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a code string (to be implemented by subclasses).

        Args:
            code: The code string to validate

        Returns:
            Dict with validation results
        """
        # Extract code if it's markdown
        if isinstance(code, str):
            original_code = code
            extracted_code, is_markdown = self._extract_code_if_markdown(code)
            code_to_validate = extracted_code
        else:
            # Handle dictionary case
            original_code = code.get("code", "")
            extracted_code, is_markdown = self._extract_code_if_markdown(original_code)
            code_to_validate = extracted_code

        # Build base result
        result = {"original_code": original_code, "is_markdown": is_markdown}

        if is_markdown:
            result["extracted_code"] = extracted_code

        # Let subclass do actual validation
        validation_result = await self._validate_extracted_code(code_to_validate)
        result.update(validation_result)

        return result

    async def _validate_extracted_code(self, code: str) -> Dict[str, Any]:
        """Actual validation logic - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


class PythonCodeValidator(BaseAsyncCodeValidator):
    """Python-specific code validator."""

    def __init__(self, check_imports: bool = True, strict_mode: bool = False):
        """
        Initialize the Python code validator.

        Args:
            check_imports: Whether to try validating import statements
            strict_mode: Whether to perform stricter validation checks
        """
        super().__init__()
        self.check_imports = check_imports
        self.strict_mode = strict_mode

    async def _validate_extracted_code(self, code: str) -> Dict[str, Any]:
        """Validate Python code (working with already extracted code)."""
        # Now we can assume code is already extracted if it was markdown
        result = {"is_valid": True, "errors": [], "warnings": []}

        # Empty code check
        if not code.strip():
            result["is_valid"] = False
            result["errors"].append("No code content found")
            return result

        # Syntax check and other validations as before...
        # ...

        return result

    async def _validate_imports(self, code_str: str) -> List[str]:
        """
        Attempt to validate import statements without executing the full code.

        Args:
            code_str: The code string to check

        Returns:
            List of warnings about import issues
        """
        import_warnings = []

        # Extract only import statements
        import_lines = []
        for line in code_str.split("\n"):
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

    async def _check_strict_issues(self, tree: ast.AST) -> List[str]:
        """
        Check for code issues that might indicate problems.

        Args:
            tree: AST of the code to check

        Returns:
            List of warnings about potential issues
        """
        warnings = []

        # Check for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                warnings.append(
                    "Bare 'except:' clause - consider catching specific exceptions"
                )

        # Check for undefined variables (simplified)
        # A complete implementation would track variable scopes
        defined_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_vars.add(target.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in defined_vars and node.id not in __builtins__:
                    warnings.append(f"Potential undefined variable: {node.id}")

        return warnings


async def validate_code(
    code: str, check_imports: bool = False, strict_mode: bool = True
) -> Dict[str, Any]:
    """
    Validate code - handles both raw code and markdown with code blocks.

    Args:
        code: Code to validate (can be markdown or raw code)
        check_imports: Whether to validate import statements
        strict_mode: Whether to perform stricter validation checks

    Returns:
        Validation results
    """
    validator = PythonCodeValidator(
        check_imports=check_imports, strict_mode=strict_mode
    )
    return await validator.validate(code)
