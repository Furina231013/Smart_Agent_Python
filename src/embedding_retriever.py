"""
Embedding Retriever module for handling document embeddings and retrieval.
Integrates with external embedding services and provides document processing.
"""

import os
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import glob
from .vector_store import VectorStore
from .config import Config


@dataclass
class Document:
    """Represents a document with content and metadata."""
    content: str
    metadata: Dict[str, Any]


class EmbeddingRetriever:
    """
    Handles document embedding and retrieval using external embedding services.
    Integrates with vector store for similarity search.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        vector_store: VectorStore = None
    ):
        """
        Initialize the embedding retriever.

        Args:
            api_key: API key for embedding service
            base_url: Base URL for embedding service
            model: Embedding model name
            vector_store: Vector store instance
        """
        self.api_key = api_key or Config.EMBEDDING_KEY
        self.base_url = base_url or Config.EMBEDDING_BASE_URL
        self.model = model or Config.EMBEDDING_MODEL
        self.vector_store = vector_store or VectorStore()

        # HTTP session for making requests
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for the given text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If embedding creation fails
        """
        session = await self._get_session()

        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Embedding API error: {response.status} - {error_text}")

                data = await response.json()
                embedding = data["data"][0]["embedding"]

                return embedding

        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to create embedding: {e}")

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Process in batches to avoid rate limits
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await asyncio.gather(
                *[self.create_embedding(text) for text in batch],
                return_exceptions=True
            )

            for embedding in batch_embeddings:
                if isinstance(embedding, Exception):
                    print(f"Error creating embedding: {embedding}")
                    # Use zero embedding as fallback
                    all_embeddings.append([0.0] * 768)  # Default embedding size
                else:
                    all_embeddings.append(embedding)

        return all_embeddings

    def load_documents_from_directory(self, directory: str) -> List[Document]:
        """
        Load documents from a directory.

        Args:
            directory: Directory path to load documents from

        Returns:
            List of loaded documents
        """
        documents = []

        # Support multiple file formats
        file_patterns = ["*.txt", "*.md", "*.json", "*.py", "*.js", "*.ts"]

        for pattern in file_patterns:
            file_paths = glob.glob(os.path.join(directory, "**", pattern), recursive=True)

            for file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()

                    if content:  # Only add non-empty documents
                        metadata = {
                            "file_path": file_path,
                            "file_name": os.path.basename(file_path),
                            "file_type": os.path.splitext(file_path)[1],
                            "file_size": len(content)
                        }

                        # Special handling for JSON files
                        if file_path.endswith('.json'):
                            try:
                                json_data = json.loads(content)
                                metadata["json_data"] = json_data
                                # Extract meaningful text from JSON
                                content = json.dumps(json_data, indent=2, ensure_ascii=False)
                            except json.JSONDecodeError:
                                pass  # Keep as plain text if not valid JSON

                        documents.append(Document(content=content, metadata=metadata))

                except Exception as e:
                    print(f"Error loading document {file_path}: {e}")

        return documents

    async def index_documents(
        self,
        documents: List[Document],
        batch_size: int = 10
    ) -> None:
        """
        Index documents by creating embeddings and storing them.

        Args:
            documents: List of documents to index
            batch_size: Batch size for processing
        """
        if not documents:
            return

        print(f"Indexing {len(documents)} documents...")

        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.content for doc in batch]

            print(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")

            try:
                embeddings = await self.create_embeddings(texts)

                # Add to vector store
                for doc, embedding in zip(batch, embeddings):
                    self.vector_store.add_item(
                        embedding=embedding,
                        document=doc.content,
                        metadata=doc.metadata
                    )

            except Exception as e:
                print(f"Error processing batch: {e}")

        print(f"Successfully indexed {self.vector_store.size()} documents")

    async def index_directory(self, directory: str) -> None:
        """
        Index all documents in a directory.

        Args:
            directory: Directory path to index
        """
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return

        documents = self.load_documents_from_directory(directory)
        await self.index_documents(documents)

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve
            threshold: Similarity threshold

        Returns:
            List of (similarity, document, metadata) tuples
        """
        if self.vector_store.size() == 0:
            print("Warning: No documents in vector store")
            return []

        # Create embedding for query
        try:
            query_embedding = await self.create_embedding(query)
        except Exception as e:
            print(f"Error creating query embedding: {e}")
            return []

        # Search for similar documents
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            threshold=threshold
        )

        return results

    async def retrieve_context(
        self,
        query: str,
        max_context_length: int = 4000,
        top_k: int = 5
    ) -> str:
        """
        Retrieve and format context for a query.

        Args:
            query: Query string
            max_context_length: Maximum length of context
            top_k: Number of documents to retrieve

        Returns:
            Formatted context string
        """
        results = await self.retrieve(query, top_k=top_k, threshold=0.5)

        if not results:
            return "No relevant context found."

        # Format results as context
        context_parts = []
        current_length = 0

        for similarity, document, metadata in results:
            # Add metadata info
            source_info = ""
            if metadata and "file_name" in metadata:
                source_info = f"[Source: {metadata['file_name']}]"

            formatted_doc = f"{source_info}\n{document}\n"

            # Check if adding this document would exceed the limit
            if current_length + len(formatted_doc) > max_context_length:
                # Truncate the document if needed
                remaining_space = max_context_length - current_length - len(source_info) - 20
                if remaining_space > 100:  # Only include if meaningful content
                    truncated_doc = document[:remaining_space] + "..."
                    formatted_doc = f"{source_info}\n{truncated_doc}\n"
                    context_parts.append(formatted_doc)
                break

            context_parts.append(formatted_doc)
            current_length += len(formatted_doc)

        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever.

        Returns:
            Dictionary with statistics
        """
        stats = self.vector_store.get_stats()
        stats.update({
            "embedding_model": self.model,
            "api_base_url": self.base_url
        })
        return stats

    async def close(self) -> None:
        """Close the embedding retriever and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()