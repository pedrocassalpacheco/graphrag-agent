from langflow.components.data import URLComponent, FileComponent
from langflow.components.processing import SplitTextComponent
from langflow.components.embeddings import OpenAIEmbeddingsComponent
from langflow.components.vectorstores import AstraDBVectorStoreComponent
from langflow.graph import Graph

def rag_data_ingestion_pipeline():
    # URL component to fetch data from websites
    url_component = URLComponent()
    url_component.urls = ["https://example.com", "https://another-example.com"]
    url_component.format = "Raw HTML"

    # File component to load PDF files
    file_component = FileComponent()
    file_component.path = "/path/to/local/pdf/files"

    # Split text component to process text from URL and files
    split_text_component = SplitTextComponent()
    split_text_component.set(data_inputs=[url_component.data, file_component.data])

    # Embeddings component to convert text to embeddings
    embeddings_component = OpenAIEmbeddingsComponent()

    # Vector store component to store embeddings
    vector_store_component = AstraDBVectorStoreComponent()
    vector_store_component.set(
        embedding_model=embeddings_component.build_embeddings,
        ingest_data=split_text_component.split_text
    )

    # Define the graph
    return Graph(start=[url_component, file_component], end=vector_store_component)

# Create and build the graph
graph = rag_data_ingestion_pipeline()
graph.build()