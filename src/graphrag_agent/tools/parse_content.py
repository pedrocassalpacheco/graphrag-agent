from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


@dataclass
class ParsedContent:
    """Standardized output format for all parsers."""

    source: str  # File path or URL
    section: str  # Class name, function name, H1/H2 heading, etc.
    content: str  # The actual content/text
    metadata: dict  # Additional parser-specific data
    content_type: str  # "class", "function", "heading", "paragraph", etc.
    vector: Optional[List[float]] = None  # Embedding vector

    def to_langchain_document(self) -> "Document":
        """
        Convert ParsedContent to LangChain Document.

        Returns:
            Document: LangChain Document with content as page_content and
                     all other fields in metadata
        """
        try:
            from langchain_core.documents import Document
        except ImportError:
            raise ImportError(
                "langchain-core is required to convert to Document. "
                "Install with: pip install langchain-core"
            )

        # Prepare metadata dictionary
        doc_metadata = self.metadata.copy()
        doc_metadata.update(
            {
                "source": self.source,
                "section": self.section,
                "content_type": self.content_type,
            }
        )

        # Add vector to metadata if present
        if self.vector is not None:
            doc_metadata["vector"] = self.vector

        return Document(page_content=self.content, metadata=doc_metadata)

    @classmethod
    def from_langchain_document(cls, doc: "Document") -> "ParsedContent":
        """
        Create ParsedContent from LangChain Document.

        Args:
            doc: LangChain Document

        Returns:
            ParsedContent: Converted ParsedContent object
        """
        # Extract our specific fields from metadata
        metadata = doc.metadata.copy()
        source = metadata.pop("source", "")
        section = metadata.pop("section", "")
        content_type = metadata.pop("content_type", "document")
        vector = metadata.pop("vector", None)

        return cls(
            source=source,
            section=section,
            content=doc.page_content,
            metadata=metadata,
            content_type=content_type,
            vector=vector,
        )
