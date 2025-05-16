from langflow.components.data import FileComponent, URLComponent
from langflow.components.embeddings import OllamaEmbeddingsComponent
from langflow.components.inputs import ChatInput
from langflow.components.models import OpenAIModelComponent
from langflow.components.outputs import ChatOutput
from langflow.components.processing import ParseDataComponent, SplitTextComponent
from langflow.components.vectorstores import OpenSearchVectorStoreComponent
from langflow.graph import Graph

def document_ingestion_flow():
    """
    Ingests documents from PDFs and websites, splits the content, embeds using Ollama, and stores in OpenSearch.
    """
    # Components for ingesting PDFs and URLs
    file_component = FileComponent()
    url_component = URLComponent()
    
    # Component to parse and split text
    parse_data_component = ParseDataComponent()
    split_text_component = SplitTextComponent()
    split_text_component.set(data_inputs=[parse_data_component.parse_data, url_component.fetch_content])

    # Embedding component
    ollama_embeddings = OllamaEmbeddingsComponent()
    ollama_embeddings.set(base_url='http://localhost:11434', model_name='all-minilm:latest')

    # OpenSearch vector store component
    opensearch_component = OpenSearchVectorStoreComponent()
    opensearch_component.set(
        opensearch_url='http://localhost:9200',
        index_name='document_chunks',
        embedding=ollama_embeddings.embeddings
    )

    # Graph definition
    return Graph(
        start=[file_component, url_component],
        end=opensearch_component
    )

def chat_with_user_flow():
    """
    Prompts user for questions about ingested documents, queries OpenSearch, and chats with the user.
    """
    # User input component
    chat_input = ChatInput()

    # Embedding component
    ollama_embeddings = OllamaEmbeddingsComponent()
    ollama_embeddings.set(base_url='http://localhost:11434', model_name='all-minilm:latest')

    # OpenSearch query component
    opensearch_query = OpenSearchVectorStoreComponent()
    opensearch_query.set(
        opensearch_url='http://localhost:9200',
        index_name='document_chunks',
        search_query=chat_input.message_response,
        embedding=ollama_embeddings.embeddings
    )

    # Parse retrieved documents
    parse_data = ParseDataComponent()
    parse_data.set(data=opensearch_query.search_documents)

    # OpenAI model for generating responses
    openai_model = OpenAIModelComponent()
    openai_model.set(input_value=parse_data.parse_data)

    # Output component
    chat_output = ChatOutput()
    chat_output.set(input_value=openai_model.text_response)

    # Graph definition
    return Graph(
        start=chat_input,
        end=chat_output
    )