from dataclasses import dataclass


@dataclass
class ParsedContent:
    """Standardized output format for all parsers."""

    source: str  # File path or URL
    section: str  # Class name, function name, H1/H2 heading, etc.
    content: str  # The actual content/text
    metadata: dict  # Additional parser-specific data
    content_type: str  # "class", "function", "heading", "paragraph", etc.
