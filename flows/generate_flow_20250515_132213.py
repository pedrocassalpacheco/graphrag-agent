from langflow.components.data import FileComponent
from langflow.components.embeddings import OllamaEmbeddingsComponent
from langflow.components.processing import SplitTextComponent
from langflow.components.vectorstores import AstraDBVectorStoreComponent
from langflow.graph import Graph

def pdf_to_astradb_embedding_flow():
    # File component to load PDF files from the "testme" folder
    file_loader = FileComponent()
    file_loader.set(path="testme/*.pdf")
    
    # Split text component to chunk the loaded PDF files
    text_splitter = SplitTextComponent()
    text_splitter.set(data_inputs=file_loader.load_files)
    
    # Ollama embeddings component to create embeddings from the text chunks
    ollama_embeddings = OllamaEmbeddingsComponent()
    ollama_embeddings.set(
        model_name="nomic-embed-text:latest",
        base_url="http://localhost:11434"  # Assuming Ollama is running locally
    )
    ollama_embeddings.set(input_text=text_splitter.split_text)
    
    # AstraDB vector store component to save the embeddings
    astradb_vector_store = AstraDBVectorStoreComponent()
    astradb_vector_store.set(
        token="ASTRA_DB_APPLICATION_TOKEN",
        environment="prod",
        database_name="your_database_name",
        api_endpoint="https://your-astradb-endpoint",
        keyspace="your_keyspace",
        collection_name="your_collection_name",
        embedding_model=ollama_embeddings.embeddings
    )
    
    # Create the graph
    return Graph(start=file_loader, end=astradb_vector_store)