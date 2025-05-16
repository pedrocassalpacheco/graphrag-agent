from langflow.components.data import FileComponent
from langflow.components.embeddings import OllamaEmbeddingsComponent
from langflow.components.processing.split_text import SplitTextComponent
from langflow.components.vectorstores import OpenSearchVectorStoreComponent
from langflow.graph import Graph

def pdf_ingestion_to_vector_store_graph():
    # Components
    file_loader = FileComponent(file_types=['pdf'])  # Load PDF files
    text_splitter = SplitTextComponent(chunk_size=500, overlap=100)  # Split text into chunks
    ollama_embeddings = OllamaEmbeddingsComponent(model='all-minilm:latest')  # Ollama embeddings model
    vector_store = OpenSearchVectorStoreComponent(
        opensearch_url='http://localhost:9200',
        index_name='pdf_vectors',
        username='admin',
        password='admin',
        use_ssl=True,
        verify_certs=False
    )  # OpenSearch vector store

    # Connections
    text_splitter.set(data_inputs=file_loader.load_files)  # Connect file loader to text splitter
    ollama_embeddings.set(text_inputs=text_splitter.split_text)  # Connect text splitter to Ollama embeddings
    vector_store.set(
        ingest_data=ollama_embeddings.embeddings,  # Connect Ollama embeddings to vector store
        embedding=ollama_embeddings  # Provide embedding function
    )

    # Graph
    return Graph(start=file_loader, end=vector_store)