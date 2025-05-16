from langflow.components.data import GitHubComponent
from langflow.components.embeddings import OllamaEmbeddingsComponent
from langflow.components.processing import ParseDataComponent
from langflow.components.vectorstores import OpenSearchVectorStoreComponent
from langflow.graph import Graph

def github_code_embeddings_flow():
    # Component to fetch source code from GitHub
    github_component = GitHubComponent(repo_url="https://github.com/your-repo-url")
    
    # Component to parse method comments from the source code
    parse_data_component = ParseDataComponent()
    parse_data_component.set(data=github_component.fetch_code)
    
    # Component to generate embeddings using Ollama
    ollama_embeddings = OllamaEmbeddingsComponent()
    ollama_embeddings.set(
        model_name="all-minilm:latest",
        base_url="http://127.0.0.1:11434"
    )
    
    # Component to save embeddings to OpenSearch
    opensearch_vector_store = OpenSearchVectorStoreComponent()
    opensearch_vector_store.set(
        opensearch_url="http://localhost:9200",
        index_name="github_code_embeddings",
        ingest_data=ollama_embeddings.embeddings
    )
    
    # Create the graph
    return Graph(
        start=github_component,
        end=opensearch_vector_store
    )