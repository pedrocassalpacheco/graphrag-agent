from langflow.components.data import URLComponent
from langflow.components.processing import ParseDataComponent
from langflow.components.vectorstores import AstraDBVectorStoreComponent
from langflow.graph import Graph

def website_data_to_rag_flow():
    # Component to fetch data from the website
    url_component = URLComponent()
    url_component.set(urls="https://yourwebsite.com", format="Raw HTML")

    # Component to parse the raw HTML data into structured Data objects
    parse_data_component = ParseDataComponent()
    parse_data_component.set(data=url_component.data)

    # Component to store the parsed data into a vector store for RAG application
    vector_store_component = AstraDBVectorStoreComponent()
    vector_store_component.set(data=parse_data_component.data)

    # Create and return the graph
    return Graph(start=url_component, end=vector_store_component)