import asyncio
import json
import httpx
from typing import List
import io
from graphrag_agent.utils.logging_config import get_logger  # Shared logger
from ollama import AsyncClient, chat
from typing import Dict, Union

logger = get_logger(__name__)  # Get a logger for this module
qa_count = 0


class BaseAsyncQuestionGenerator:
    """
    Base class for asynchronous question generation using an LLM.
    Provides common functionality for querying an LLM and processing a queue.
    """

    def __init__(self, model: str):
        self.model = model
        self.qa_count: int = 0

    async def query_llm(self, title: str, content: str) -> List[str]:
        """
        Abstract method to query the LLM. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the query_llm method.")

    async def generate_qa(
        self,
        input: asyncio.Queue,
        output: Union[
            asyncio.Queue, Dict[str, Dict[str, List[str]]], io.TextIOWrapper
        ] = None,
    ):
        """
        Continuously processes items from the queue, queries the LLM, and writes results to a file.
        Accepts an already open file handle instead of a file name.
        """
        while True:
            item = await input.get()
            if item is None:  # Signal that processing is complete
                logger.info("QA generator received termination signal, finishing")
                if isinstance(output, io.TextIOWrapper):
                    output.close()
                    break

            url = item.get("url")
            title = item.get("title")
            content = self.format_content_list(
                item.get("content")
            )  # Format content list

            logger.info(f"Using {self.model} to create Q&A for {url} {title}")

            questions = await self.query_llm(title, content)
            self.qa_count += 1

            logger.info(f"Q&A {self.qa_count}: {questions}")

            if isinstance(output, asyncio.Queue):
                await output.put({"url": questions})

            elif isinstance(output, dict):
                output[url] = questions
            elif isinstance(output, io.TextIOWrapper):
                # Write to the file handle
                output.write(
                    json.dumps({"url": url, "title": title, "questions": questions})
                    + "\n"
                )

            input.task_done()  # Mark the item as processed

    def format_content_list(self, content_list: List) -> str:
        formatted_text = ""
        for item in content_list:
            # Skip empty strings
            if item == "":
                continue

            # Handle image dictionaries
            if isinstance(item, dict) and "image" in item:
                continue
            elif isinstance(item, dict) and "table" in item:
                continue
            elif isinstance(item, str):
                formatted_text += item + "\n\n"

        return formatted_text.strip()  # Remove trailing newline


class AsyncQuestionGenerator1(BaseAsyncQuestionGenerator):
    """
    Asynchronous tool to generate questions using a local LLM via HTTP requests.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/completions",
        model: str = "llama3.3:lattest",
    ):
        super().__init__(model)
        self.ollama_url = ollama_url

    async def query_llm(self, title: str, content: str) -> List[str]:
        """
        Sends a prompt to the local LLM via HTTP and retrieves the generated questions.
        """
        prompt = f"""
        Based on the following title and content, generate a list of questions that can be answered:
        Title: {title}
        Content: {content}
        Format the output as an Alpaca-compatible list for future model tuning.
        """
        payload = {"model": self.model, "prompt": prompt}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.ollama_url, json=payload)
                response.raise_for_status()
                questions = response.json().get("response", "[]")
                return json.loads(questions)  # Parse the JSON list of questions
            except Exception as e:
                logger.error(f"Error querying LLM: {e}")
                return []


class AsyncQuestionGenerator2(BaseAsyncQuestionGenerator):
    """
    Asynchronous tool to generate questions using a local LLM via the Ollama Python client.
    """

    def __init__(self, model: str = "llama2"):
        super().__init__(model)
        self.client = chat  # Initialize the Ollama client

    async def query_llm(self, title: str, content: str) -> List[str]:
        """
        Sends a prompt to the local LLM via the Ollama Python client and retrieves the generated questions.
        """
        prompt = f"""
        Based on the following title and content, generate a list of questions that can be answered and the respective answers using just the title and content
        Title: {title}
        Content: {content}
        Structured the output according the Alpaca-compatible list for future model tuning.
        """
        try:
            # Use the Ollama client to query the model
            response = self.client(model=self.model, prompt=prompt)
            questions = json.loads(
                response.get("response", "[]")
            )  # Parse the JSON list of questions
            return questions
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return []


class AsyncQuestionGenerator3(BaseAsyncQuestionGenerator):
    """
    Asynchronous tool to generate questions using a local LLM via the Ollama Python client with async calls.
    """

    def __init__(self, model: str = "llama2"):
        super().__init__(model)
        self.client = AsyncClient()  # Async client for Ollama

    async def query_llm(self, title: str, content: str) -> List[str]:
        """
        Sends a prompt to the local LLM via the Ollama Python client and retrieves the generated questions asynchronously.
        """
        prompt = f"""
        Based on the following title and content, generate a question and answer list containing the question and and answer
        Title: {title}
        Content: {content}
        Format the output as a JSON list of strings, where each string is a question.
        """

        try:
            response = await AsyncClient().chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return response.message.content

        except Exception as e:
            logger.error(f"Error querying LLM asynchronously: {e}")
            return []
