from langflow.components.data import FileComponent
from langflow.components.embeddings import OllamaEmbeddingsComponent
from langflow.components.inputs import ChatInput
from langflow.components.models import OpenAIModelComponent
from langflow.components.outputs import ChatOutput
from langflow.components.processing import ParseDataComponent
from langflow.components.vectorstores import OpenSearchVectorStoreComponent
from langflow.graph import Graph

def rag_flow_with_pdf_and_ollama():
    # Components
    file_component = FileComponent()
    parse_data = ParseDataComponent()
    ollama_embeddings = OllamaEmbeddingsComponent()
    opensearch_vector_store = OpenSearchVectorStoreComponent()
    chat_input = ChatInput()
    openai_model = OpenAIModelComponent()
    chat_output = ChatOutput()

    # Set up the flow
    parse_data.set(data=file_component.load_files)
    ollama_embeddings.set(input_data=parse_data.parse_data)
    opensearch_vector_store.set(embedding=ollama_embeddings.embeddings)
    openai_model.set(input_value=chat_input.message_response)
    chat_output.set(input_value=openai_model.text_response)

    # Create the graph
    return Graph(start=file_component, end=chat_output)

# Example usage
graph = rag_flow_with_pdf_and_ollama()
graph.build()