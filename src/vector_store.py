"""
Vector Store module for handling high-dimensional vector operations.
Implements in-memory vector storage with cosine similarity search.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pickle
import os


@dataclass
class VectorStoreItem:
    """Represents an item stored in the vector store."""
    embedding: np.ndarray
    document: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Convert embedding to numpy array if needed."""
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)

        if self.metadata is None:
            self.metadata = {}


class VectorStore:
    """
    In-memory vector store with cosine similarity search.
    Supports adding, searching, and persisting vector items.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the vector store.

        Args:
            max_size: Maximum number of items to store
        """
        self.items: List[VectorStoreItem] = []
        self.max_size = max_size
        self._embedding_matrix: Optional[np.ndarray] = None

    def add_item(self, embedding: List[float], document: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add an item to the vector store.

        Args:
            embedding: Vector embedding
            document: Document text
            metadata: Optional metadata
        """
        item = VectorStoreItem(
            embedding=np.array(embedding, dtype=np.float32),
            document=document,
            metadata=metadata or {}
        )

        # Remove oldest item if at capacity
        if len(self.items) >= self.max_size:
            self.items.pop(0)

        self.items.append(item)
        self._embedding_matrix = None  # Reset cache

    def add_items(self, items: List[Tuple[List[float], str, Dict[str, Any]]]) -> None:
        """
        Add multiple items to the vector store.

        Args:
            items: List of (embedding, document, metadata) tuples
        """
        for embedding, document, metadata in items:
            self.add_item(embedding, document, metadata)

    def _get_embedding_matrix(self) -> np.ndarray:
        """
        Get all embeddings as a matrix.

        Returns:
            Matrix of shape (n_items, embedding_dim)
        """
        if self._embedding_matrix is None or len(self._embedding_matrix) != len(self.items):
            if not self.items:
                self._embedding_matrix = np.array([]).reshape(0, 0)
            else:
                self._embedding_matrix = np.stack([item.embedding for item in self.items])

        return self._embedding_matrix

    def _cosine_similarity(self, query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and all embeddings.

        Args:
            query_embedding: Query vector
            embeddings: Matrix of embeddings

        Returns:
            Array of similarity scores
        """
        if embeddings.size == 0:
            return np.array([])

        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

        # Calculate cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)
        return similarities

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[float, str, Dict[str, Any]]]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (similarity, document, metadata) tuples sorted by similarity
        """
        if not self.items:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        embedding_matrix = self._get_embedding_matrix()

        # Calculate similarities
        similarities = self._cosine_similarity(query_vec, embedding_matrix)

        # Get indices of top results above threshold
        valid_indices = np.where(similarities >= threshold)[0]

        if len(valid_indices) == 0:
            return []

        # Sort by similarity and take top_k
        sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1][:top_k]]

        # Prepare results
        results = []
        for idx in sorted_indices:
            item = self.items[idx]
            results.append((
                float(similarities[idx]),
                item.document,
                item.metadata.copy()
            ))

        return results

    def get_all_documents(self) -> List[str]:
        """
        Get all documents in the store.

        Returns:
            List of all documents
        """
        return [item.document for item in self.items]

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        """
        Get all metadata in the store.

        Returns:
            List of all metadata dictionaries
        """
        return [item.metadata.copy() for item in self.items]

    def size(self) -> int:
        """
        Get the number of items in the store.

        Returns:
            Number of stored items
        """
        return len(self.items)

    def clear(self) -> None:
        """Clear all items from the store."""
        self.items.clear()
        self._embedding_matrix = None

    def remove_item(self, index: int) -> None:
        """
        Remove an item by index.

        Args:
            index: Index of the item to remove
        """
        if 0 <= index < len(self.items):
            self.items.pop(index)
            self._embedding_matrix = None

    def save(self, filepath: str) -> None:
        """
        Save the vector store to a file.

        Args:
            filepath: Path to save the store
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert numpy arrays to lists for pickling
        serializable_items = []
        for item in self.items:
            serializable_items.append({
                'embedding': item.embedding.tolist(),
                'document': item.document,
                'metadata': item.metadata
            })

        with open(filepath, 'wb') as f:
            pickle.dump({
                'items': serializable_items,
                'max_size': self.max_size
            }, f)

    def load(self, filepath: str) -> None:
        """
        Load the vector store from a file.

        Args:
            filepath: Path to load the store from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vector store file not found: {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.max_size = data['max_size']
        self.items.clear()

        for item_data in data['items']:
            item = VectorStoreItem(
                embedding=np.array(item_data['embedding'], dtype=np.float32),
                document=item_data['document'],
                metadata=item_data['metadata']
            )
            self.items.append(item)

        self._embedding_matrix = None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with store statistics
        """
        if not self.items:
            return {
                'size': 0,
                'embedding_dim': 0,
                'avg_document_length': 0
            }

        embedding_matrix = self._get_embedding_matrix()
        avg_doc_length = np.mean([len(item.document) for item in self.items])

        return {
            'size': len(self.items),
            'embedding_dim': embedding_matrix.shape[1] if embedding_matrix.size > 0 else 0,
            'avg_document_length': float(avg_doc_length),
            'max_size': self.max_size,
            'utilization': len(self.items) / self.max_size
        }