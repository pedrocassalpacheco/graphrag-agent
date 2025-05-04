import asyncio
import json
import httpx
from typing import List, Optional
from utils.logging_config import get_logger  # Shared logger
from ollama import Ollama  # Import the Ollama Python client


logger = get_logger(__name__)  # Get a logger for this module


class AsyncQuestionGeneratorv1:
    """
    Asynchronous tool to generate questions using a local LLM via Ollama.
    This tool takes URLs, titles, and content from a queue and prompts the LLM
    to generate all possible questions that can be answered by the content.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/completions",
        model: str = "llama2",
    ):
        self.ollama_url = ollama_url
        self.model = model

    async def query_llm(self, title: str, content: str) -> List[str]:
        """
        Sends a prompt to the local LLM via Ollama and retrieves the generated questions.
        """
        prompt = f"""
        Based on the following title and content, generate a list of questions that can be answered:
        Title: {title}
        Content: {content}
        Format the output as a JSON list of strings, where each string is a question.
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

    async def process_queue(
        self, queue: asyncio.Queue, output_file: Optional[str] = None
    ):
        """
        Continuously processes items from the queue, queries the LLM, and writes results to a file.
        """
        async with (
            open(output_file, "a", encoding="utf-8")
            if output_file
            else None as file_handle
        ):
            while True:
                item = await queue.get()
                if item is None:  # Signal that processing is complete
                    break

                url = item.get("url")
                title = item.get("title")
                content = " ".join(
                    item.get("content", [])
                )  # Combine content into a single string

                logger.info(f"Generating questions for URL: {url}, Title: {title}")
                questions = await self.query_llm(title, content)

                # Write results to the file if a file handle is provided
                if file_handle:
                    result = {"url": url, "title": title, "questions": questions}
                    file_handle.write(json.dumps(result) + "\n")

                queue.task_done()  # Mark the task as done


class AsyncQuestionGeneratorv2:
    """
    Asynchronous tool to generate questions using a local LLM via the Ollama Python client.
    This tool takes URLs, titles, and content from a queue and prompts the LLM
    to generate all possible questions that can be answered by the content.
    """

    def __init__(self, model: str = "llama2"):
        self.model = model
        self.client = Ollama()  # Initialize the Ollama client

    async def query_llm(self, title: str, content: str) -> List[str]:
        """
        Sends a prompt to the local LLM via the Ollama Python client and retrieves the generated questions.
        """
        prompt = f"""
        Based on the following title and content, generate a list of questions that can be answered:
        Title: {title}
        Content: {content}
        Format the output as a JSON list of strings, where each string is a question.
        """
        try:
            # Use the Ollama client to query the model
            response = self.client.generate(model=self.model, prompt=prompt)
            questions = json.loads(
                response.get("response", "[]")
            )  # Parse the JSON list of questions
            return questions
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return []

    async def process_queue(
        self, queue: asyncio.Queue, output_file: Optional[str] = None
    ):
        """
        Continuously processes items from the queue, queries the LLM, and writes results to a file.
        """
        async with (
            open(output_file, "a", encoding="utf-8")
            if output_file
            else None as file_handle
        ):
            while True:
                item = await queue.get()
                if item is None:  # Signal that processing is complete
                    break

                url = item.get("url")
                title = item.get("title")
                content = " ".join(
                    item.get("content", [])
                )  # Combine content into a single string

                logger.info(f"Generating questions for URL: {url}, Title: {title}")
                questions = await self.query_llm(title, content)

                # Write results to the file if a file handle is provided
                if file_handle:
                    result = {"url": url, "title": title, "questions": questions}
                    file_handle.write(json.dumps(result) + "\n")

                queue.task_done()  # Mark the task as done
