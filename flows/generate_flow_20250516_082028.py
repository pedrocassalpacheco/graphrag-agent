from langflow.components.data import FileComponent
from langflow.components.embeddings import OllamaEmbeddingsComponent
from langflow.components.processing import SplitTextComponent
from langflow.components.vectorstores import OpenSearchVectorStoreComponent
from langflow.graph import Graph

def ingestion_graph():
    """
    Creates a document ingestion graph using Ollama embeddings and OpenSearch vector store.
    
    This function builds a graph that:
    1. Loads files using a FileComponent
    2. Splits the text into chunks with SplitTextComponent
    3. Creates embeddings using Ollama's embedding model
    4. Stores the embeddings in an OpenSearch vector store
    
    Returns:
        Graph: A Langflow Graph object with file_component as start and vector_store as end
    """
    file_component = FileComponent()
    text_splitter = SplitTextComponent()
    text_splitter.set(data_inputs=file_component.load_files)
    ollama_embeddings = OllamaEmbeddingsComponent()
    ollama_embeddings.set(model_name='all-minilm:latest', base_url='http://localhost:11434')
    vector_store = OpenSearchVectorStoreComponent()
    vector_store.set(
        opensearch_url='http://localhost:9200',
        index_name='langflow',
        ingest_data=text_splitter.split_text,
        embedding=ollama_embeddings.embeddings
    )

    return Graph(start=file_component, end=vector_store)

def retrieval_graph():
    """
    Creates a retrieval graph using OpenSearch vector store.
    
    This function builds a graph that:
    1. Takes user input and searches it in the OpenSearch vector store
    2. Outputs the search results
    
    Returns:
        Graph: A Langflow Graph object with chat_input as start and chat_output as end
    """
    from langflow.components.inputs import ChatInput
    from langflow.components.outputs import ChatOutput

    chat_input = ChatInput()
    vector_store = OpenSearchVectorStoreComponent()
    vector_store.set(
        opensearch_url='http://localhost:9200',
        index_name='langflow',
        search_input=chat_input.message_response
    )
    chat_output = ChatOutput()
    chat_output.set(input_value=vector_store.search_results)

    return Graph(start=chat_input, end=chat_output)