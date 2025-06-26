import asyncio
import os
from typing import List, Dict, Any
import dotenv
from graphrag_agent.utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGSystem:
    """Simple RAG system using existing FAISS vector store."""

    def __init__(
        self,
        faiss_index_dir: str = "index/",
        api_key: str = None,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4",
        max_context_docs: int = 5,
    ):
        self.faiss_index_dir = faiss_index_dir
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.max_context_docs = max_context_docs
        self.vectorstore = None
        self.openai_client = None
        self._initialized = False

    async def initialize(self):
        """Initialize the RAG system by loading existing FAISS index."""
        try:
            logger.info("Initializing RAG system...")

            # Check if FAISS index directory exists
            if not os.path.exists(self.faiss_index_dir):
                raise FileNotFoundError(
                    f"FAISS index directory not found: {self.faiss_index_dir}"
                )

            # Initialize OpenAI client for LLM
            from openai import AsyncOpenAI

            self.openai_client = AsyncOpenAI(api_key=self.api_key)

            # Load FAISS vector store
            await self.load_faiss_index()

            logger.info("RAG system initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise

    async def load_faiss_index(self):
        """Load existing FAISS index from disk."""
        try:
            logger.info(f"Loading FAISS index from {self.faiss_index_dir}...")

            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings

            # Initialize the same embedding model that was used to create the index
            embedding_model = OpenAIEmbeddings(
                api_key=self.api_key,
                model=self.embedding_model,
                dimensions=3072,  # Must match the dimensions used when creating the index
            )

            # Load FAISS vector store
            self.vectorstore = await asyncio.to_thread(
                FAISS.load_local,
                self.faiss_index_dir,
                embedding_model,
                allow_dangerous_deserialization=True,  # Required for loading
            )

            self._initialized = True
            logger.info("FAISS index loaded successfully")

            # Get some stats about the loaded index
            try:
                # Test query to get index size info
                test_results = await asyncio.to_thread(
                    self.vectorstore.similarity_search, "test", k=1
                )
                logger.info(f"FAISS index appears to contain documents")
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")

        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise

    async def search_similar_documents(
        self, query: str, k: int = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector store."""
        if not self._initialized:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")

        k = k or self.max_context_docs

        try:
            # Perform similarity search
            results = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_score, query, k=k
            )

            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity_score": float(score),
                    }
                )

            logger.info(
                f"Found {len(formatted_results)} similar documents for query: {query[:50]}..."
            )
            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc["metadata"].get("source", "Unknown")
            section = doc["metadata"].get("section", "")
            content = doc["content"]
            score = doc.get("similarity_score", 0)

            context_part = f"""
Document {i} (Source: {source}, Section: {section}, Score: {score:.3f}):
{content}
"""
            context_parts.append(context_part.strip())

        return "\n\n" + "\n\n".join(context_parts)

    async def generate_answer(
        self, query: str, context_documents: List[Dict[str, Any]]
    ) -> str:
        """Generate answer using LLM with retrieved context."""
        try:
            # Build context from retrieved documents
            context = self._build_context(context_documents)

            # Create prompt
            prompt = f"""You are a helpful AI assistant. Answer the user's question based on the provided context documents.

Context Documents:
{context}

User Question: {query}

Instructions:
- Answer based primarily on the provided context
- If the context doesn't contain enough information, say so
- Cite which document(s) you're referencing when possible
- Be concise but thorough

Answer:"""

            # Generate response
            response = await self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that answers questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.1,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return (
                f"Sorry, I encountered an error while generating the answer: {str(e)}"
            )

    async def rag_query(self, query: str) -> Dict[str, Any]:
        """Perform complete RAG query: retrieve + generate."""
        try:
            logger.info(f"Processing RAG query: {query}")

            # Step 1: Retrieve similar documents
            similar_docs = await self.search_similar_documents(query)

            if not similar_docs:
                return {
                    "query": query,
                    "answer": "I couldn't find any relevant documents to answer your question.",
                    "sources": [],
                    "context_used": False,
                }

            # Step 2: Generate answer with context
            answer = await self.generate_answer(query, similar_docs)

            # Step 3: Format response
            sources = [
                {
                    "source": doc["metadata"].get("source", "Unknown"),
                    "section": doc["metadata"].get("section", ""),
                    "similarity_score": doc.get("similarity_score", 0),
                }
                for doc in similar_docs
            ]

            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "context_used": True,
                "num_sources": len(similar_docs),
            }

        except Exception as e:
            logger.error(f"Error in RAG query: {str(e)}")
            return {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "context_used": False,
            }

    async def get_index_info(self) -> Dict[str, Any]:
        """Get information about the loaded FAISS index."""
        if not self._initialized:
            await self.initialize()

        try:
            # Try to get some basic info about the index
            sample_docs = await asyncio.to_thread(
                self.vectorstore.similarity_search, "sample query", k=5
            )

            return {
                "index_directory": self.faiss_index_dir,
                "sample_documents_found": len(sample_docs),
                "sample_sources": list(
                    set([doc.metadata.get("source", "Unknown") for doc in sample_docs])
                ),
                "initialized": self._initialized,
            }

        except Exception as e:
            logger.error(f"Error getting index info: {str(e)}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup resources."""
        if self.openai_client:
            await self.openai_client.close()
        logger.info("RAG system cleanup completed")


async def main():
    """Main demo function."""
    # Load environment variables
    dotenv.load_dotenv()

    # Configuration
    FAISS_INDEX_DIR = os.getenv("FAISS_DIR", "index/")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    try:
        # Initialize RAG system
        rag = RAGSystem(
            faiss_index_dir=FAISS_INDEX_DIR,
            api_key=OPENAI_API_KEY,
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4",
            max_context_docs=5,
        )

        await rag.initialize()

        # Get index information
        index_info = await rag.get_index_info()
        logger.info(f"Index info: {index_info}")

        # Sample queries for demonstration
        sample_queries = [
            "What is LangFlow?",
            "How do I create custom components?",
            "What are the main features of LangFlow?",
            "How does the vector store work?",
            "What Python libraries does LangFlow use?",
            "What components are there from chat completion?",
        ]

        logger.info("Starting RAG demonstration...")

        for query in sample_queries:
            logger.info(f"\n{'='*60}")
            logger.info(f"Query: {query}")
            logger.info(f"{'='*60}")

            # Perform RAG query
            result = await rag.rag_query(query)

            # Display results
            print(f"\nQuery: {result['query']}")
            print(f"\nAnswer:\n{result['answer']}")

            if result["context_used"]:
                print(f"\nSources used ({result['num_sources']} documents):")
                for i, source in enumerate(result["sources"], 1):
                    print(
                        f"  {i}. {source['source']} - {source['section']} (score: {source['similarity_score']:.3f})"
                    )
            else:
                print("\nNo relevant sources found.")

            print("\n" + "-" * 60)

            # Wait a bit between queries to avoid rate limits
            await asyncio.sleep(2)

        # Interactive mode
        logger.info("\nEntering interactive mode. Type 'quit' to exit.")
        interactive = False
        while interactive:
            try:
                user_query = input("\nEnter your question: ").strip()
                if user_query.lower() in ["quit", "exit", "q"]:
                    break

                if not user_query:
                    continue

                result = await rag.rag_query(user_query)

                print(f"\nAnswer:\n{result['answer']}")

                if result["context_used"]:
                    print(f"\nSources ({result['num_sources']} documents):")
                    for i, source in enumerate(result["sources"], 1):
                        print(
                            f"  {i}. {source['source']} - {source['section']} (score: {source['similarity_score']:.3f})"
                        )

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")

        # Cleanup
        await rag.cleanup()
        logger.info("RAG demo completed")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
