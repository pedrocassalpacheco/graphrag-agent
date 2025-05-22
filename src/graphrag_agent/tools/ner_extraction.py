import asyncio
import json
from typing import List, Dict, Any, Union, Optional
import io

import spacy
from spacy.language import Language

from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class BaseAsyncNERExtractor:
    """
    Base class for asynchronous Named Entity Recognition extraction.
    Provides common functionality for extracting entities and processing a queue.
    """

    def __init__(self, model: str):
        self.model = model
        self.extraction_count: int = 0

    async def extract_entities(
        self, text: str, doc_id: str = None
    ) -> Dict[str, List[str]]:
        """
        Abstract method to extract entities from text. Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement the extract_entities method."
        )

    async def process_queue(
        self,
        input_queue: asyncio.Queue,
        output_queue: Union[
            asyncio.Queue, Dict[str, Dict[str, List[str]]], io.TextIOWrapper
        ] = None,
    ):
        """
        Continuously processes items from the queue, extracts entities, and handles results.
        """
        while True:
            item = await input_queue.get()
            if item is None:  # Signal that processing is complete
                logger.info("NER extractor received termination signal, finishing")
                if isinstance(output_queue, io.TextIOWrapper):
                    output_queue.close()
                    break

            doc_id = item.get("id", None)
            text = item.get("text", "")
            title = item.get("title", "")

            if not text:
                logger.warning(f"Skipping empty text for document {doc_id}")
                input_queue.task_done()
                continue

            logger.info(
                f"Using {self.model} to extract entities from {doc_id or title}"
            )

            entities = await self.extract_entities(text, doc_id)
            self.extraction_count += 1

            logger.info(
                f"NER extraction {self.extraction_count} complete: {len(entities)} entity types found"
            )

            result = {"id": doc_id, "title": title, "entities": entities}

            if isinstance(output_queue, asyncio.Queue):
                await output_queue.put(result)
            elif isinstance(output_queue, dict):
                output_queue[doc_id or title] = entities
            elif isinstance(output_queue, io.TextIOWrapper):
                output_queue.write(json.dumps(result) + "\n")

            input_queue.task_done()


class SpacyNERExtractor(BaseAsyncNERExtractor):
    """
    NER extraction implementation using spaCy models.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        super().__init__(model)
        self.nlp = None

    async def load_model(self):
        """Load the spaCy model if not already loaded."""
        if self.nlp is None:
            # Run in an executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            self.nlp = await loop.run_in_executor(None, self._load_spacy_model)
            logger.info(f"Loaded spaCy model: {self.model}")

    def _load_spacy_model(self) -> Language:
        """Helper to load spaCy model synchronously."""
        try:
            return spacy.load(self.model)
        except OSError:
            logger.warning(f"Model {self.model} not found. Downloading...")
            spacy.cli.download(self.model)
            return spacy.load(self.model)

    async def extract_entities(
        self, text: str, doc_id: str = None
    ) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.
        Returns a dictionary mapping entity types to lists of entity texts.
        """
        await self.load_model()

        # Process the document in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(None, self.nlp, text)

        # Group entities by type
        entities = {}
        for ent in doc.ents:
            entity_type = ent.label_
            if entity_type not in entities:
                entities[entity_type] = []

            # Avoid duplicates
            if ent.text not in entities[entity_type]:
                entities[entity_type].append(ent.text)

        return entities


class TransformersNERExtractor(BaseAsyncNERExtractor):
    """
    NER extraction implementation using HuggingFace Transformers.
    """

    def __init__(self, model: str = "dslim/bert-base-NER"):
        super().__init__(model)
        self.tokenizer = None
        self.ner_model = None

    async def load_model(self):
        """Load the transformers model if not already loaded."""
        if self.ner_model is None:
            from transformers import (
                AutoTokenizer,
                AutoModelForTokenClassification,
                pipeline,
            )

            # Run in an executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()

            async def _load():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
                self.ner_model = AutoModelForTokenClassification.from_pretrained(
                    self.model
                )
                self.pipeline = pipeline(
                    "ner",
                    model=self.ner_model,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                )

            await loop.run_in_executor(None, lambda: _load())
            logger.info(f"Loaded Transformers model: {self.model}")

    async def extract_entities(
        self, text: str, doc_id: str = None
    ) -> Dict[str, List[str]]:
        """
        Extract named entities from text using Transformers.
        Returns a dictionary mapping entity types to lists of entity texts.
        """
        await self.load_model()

        # Process the document in an executor to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: self.pipeline(text))

        # Group entities by type
        entities = {}
        for result in results:
            entity_type = result["entity_group"]
            entity_text = result["word"]

            if entity_type not in entities:
                entities[entity_type] = []

            # Avoid duplicates
            if entity_text not in entities[entity_type]:
                entities[entity_type].append(entity_text)

        return entities


# Example usage
async def main():
    # Set up queues
    input_queue = asyncio.Queue()
    output_queue = asyncio.Queue()

    # Create extractor instance
    extractor = SpacyNERExtractor()

    # Add sample document to queue
    sample_doc = {
        "id": "doc1",
        "title": "Apple Announces New iPhone",
        "text": "Apple Inc. announced the release of its new iPhone model in Cupertino, California yesterday. Tim Cook, the CEO of Apple, presented the new features at the Steve Jobs Theater.",
    }
    await input_queue.put(sample_doc)
    await input_queue.put(None)  # Signal end of processing

    # Start processing
    await extractor.process_queue(input_queue, output_queue)

    # Print results
    result = await output_queue.get()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
