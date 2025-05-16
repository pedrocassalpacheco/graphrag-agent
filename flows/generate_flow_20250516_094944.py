from langflow.components.embeddings import OllamaEmbeddingsComponent
from langflow.components.inputs import ChatInput
from langflow.components.models import OpenAIModelComponent
from langflow.components.outputs import ChatOutput
from langflow.components.vectorstores import AstraDBVectorStoreComponent
from langflow.graph import Graph

def interactive_chat_rag_flow():
    # Components
    chat_input = ChatInput()
    ollama_embeddings = OllamaEmbeddingsComponent()
    ollama_embeddings.set(model_name='nomic', base_url='http://localhost:11434')
    
    astra_db_vector_store = AstraDBVectorStoreComponent()
    astra_db_vector_store.set(
        token='ASTRA_DB_APPLICATION_TOKEN',
        api_endpoint='ASTRA_DB_API_ENDPOINT',
        database_name='your_database_name',
        collection_name='your_collection_name',
        embedding_model=ollama_embeddings.embeddings
    )
    
    openai_model = OpenAIModelComponent()
    openai_model.set(input_value=astra_db_vector_store.search_results)
    
    chat_output = ChatOutput()
    chat_output.set(input_value=openai_model.model)

    # Graph
    return Graph(start=chat_input, end=chat_output)

# Example of building and running the graph
graph = interactive_chat_rag_flow()
graph.build()