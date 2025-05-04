import asyncio
import jsonx
import httpxort BeautifulSoup
from typing import List, Optional, Union, Dicttional, Union
from utils.logging_config import get_logger  # Shared logger
from ollama import Ollama  # Import the Ollama Python client
import json

logger = get_logger(__name__)  # Get a logger for this modulered logger

logger = get_logger(__name__)  # Get a logger for this module
class AsyncQuestionGeneratorv1:
    """
    Asynchronous tool to generate questions using a local LLM via Ollama.
    This tool takes URLs, titles, and content from a queue and prompts the LLM
    to generate all possible questions that can be answered by the content.ured content
    """m HTML elements such as headings and paragraphs.
    Attributes:
    def __init__(oat): Delay in seconds between requests to avoid overwhelming the server.
        self,
        ollama_url: str = "http://localhost:11434/api/completions",
        model: str = "llama2",float = 1.0):
    ):  self.delay = delay
        self.ollama_url = ollama_url
        self.model = modelext: str) -> str:
        """
    async def query_llm(self, title: str, content: str) -> List[str]:ols from the text.
        """
        Sends a prompt to the local LLM via Ollama and retrieves the generated questions.
        """
        prompt = f"""age(self, url: str, client: httpx.AsyncClient) -> str:
        Based on the following title and content, generate a list of questions that can be answered:
        Title: {title}{
        Content: {content}t": "Mozilla/5.0 (compatible; GraphRagParser/1.0; +https://github.com/yourusername/graphrag)"
        Format the output as a JSON list of strings, where each string is a question.
        """ response = await client.get(url, headers=headers, timeout=10.0)
        payload = {"model": self.model, "prompt": prompt}
            return response.text
        async with httpx.AsyncClient() as client:
            try:t(f"Error fetching {url}: {str(e)}")
                response = await client.post(self.ollama_url, json=payload)
                response.raise_for_status()
                questions = response.json().get("response", "[]")
                return json.loads(questions)  # Parse the JSON list of questions
            except Exception as e:
                logger.error(f"Error querying LLM: {e}"), List[str]]:
                return []oup(html, "html.parser")
        content = {}
    async def process_queue(
        self, queue: asyncio.Queue, output: Union[Dict[str, List[Dict[str, List[str]]]], io.TextIOWrapper] = None
    ):  # Ugly but effective way to parse the content
        """ element in soup.find_all(
        Continuously processes items from the queue, queries the LLM, and writes results to the output.
        The output can be a dictionary or a file pointer.     "h1",
        """,
        while True:
            item = await queue.get()
            if item is None:  # Signal that processing is complete
                break      "h6",

            url = item.get("url")
            title = item.get("title")
            content = " ".join(item.get("content", []))  # Combine content into a single string
                "blockquote",
            logger.info(f"Generating questions for URL: {url}, Title: {title}")
            questions = await self.query_llm(title, content)

            # Handle different output types
            if isinstance(output, dict):
                if url not in output:            logger.debug(
                    output[url] = []p=True)}"
                output[url].append({"title": title, "questions": questions})
            elif isinstance(output, io.TextIOWrapper):            if element.name.startswith("h"):  # Titles and subtitles
                result = {"url": url, "title": title, "questions": questions}rue))
                output.write(json.dumps(result) + "\n")_title] = []

            queue.task_done()  # Mark the task as done
                    self.clean_text(element.get_text(strip=True))

class AsyncQuestionGeneratorv2:            elif current_title and element.name in ["ul", "ol"]:  # Lists
    """                list_items = [
    Asynchronous tool to generate questions using a local LLM via the Ollama Python client.text(li.get_text(strip=True))
    This tool takes URLs, titles, and content from a queue and prompts the LLM             for li in element.find_all("li")
    to generate all possible questions that can be answered by the content.
    """

    def __init__(self, model: str = "llama2"):         rows = []
        self.model = model                for row in element.find_all("tr"):
        self.client = Ollama()  # Initialize the Ollama client
lf.clean_text(cell.get_text(strip=True))
    async def query_llm(self, title: str, content: str) -> List[str]:
        """                    ]
        Sends a prompt to the local LLM via the Ollama Python client and retrieves the generated questions.
        """     content[current_title].append({"table": rows})
        prompt = f"""
        Based on the following title and content, generate a list of questions that can be answered:     content[current_title].append(
        Title: {title}elf.clean_text(element.get_text(strip=True))
        Content: {content}
        Format the output as a JSON list of strings, where each string is a question.nt_title and element.name in ["pre", "code"]:  # Code blocks
        """rrent_title].append(
        try:
            # Use the Ollama client to query the model     )
            response = self.client.generate(model=self.model, prompt=prompt)elif current_title and element.name == "img":  # Images
            questions = json.loads(("alt", "No description"))
                response.get("response", "[]")
            )  # Parse the JSON list of questionsle].append({"image": {"alt": alt_text, "src": src}})
            return questions
        except Exception as e:o the event loop
            logger.error(f"Error querying LLM: {e}")
            return []

    async def process_queue(
        self, queue: asyncio.Queue, output: Union[Dict[str, List[Dict[str, List[str]]]], io.TextIOWrapper] = None        input: asyncio.Queue,
    ):
        """Wrapper
        Continuously processes items from the queue, queries the LLM, and writes results to the output.  ] = None,
        The output can be a dictionary or a file pointer.put_file: io.TextIOWrapper = None,
        """
        while True:nc with httpx.AsyncClient() as client:
            item = await queue.get()ue:
            if item is None:  # Signal that processing is complete
                breakNone:  # Crawling is complete

            url = item.get("url")      logger.info(f"Parsing page {url}")
            title = item.get("title")_url = self.clean_text(url)  # Clean the URL
            content = " ".join(item.get("content", []))  # Combine content into a single stringpage(cleaned_url, client)

            logger.info(f"Generating questions for URL: {url}, Title: {title}")d_content = await self.parse_content(html)
            questions = await self.query_llm(title, content)                    # Handle different output types
put, asyncio.Queue):
            # Handle different output types
            if isinstance(output, dict): cleaned_url, "parsed_content": parsed_content}
                if url not in output:
                    output[url] = []
                output[url].append({"title": title, "questions": questions})                        output[cleaned_url] = parsed_content
            elif isinstance(output, io.TextIOWrapper):
                result = {"url": url, "title": title, "questions": questions}ms():
                output.write(json.dumps(result) + "\n")                            json_line = {

            queue.task_done()  # Mark the task as done "title": title,
