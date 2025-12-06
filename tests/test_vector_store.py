"""
Tests for the VectorStore module.
"""

import pytest
import numpy as np
from src.vector_store import VectorStore, VectorStoreItem


class TestVectorStore:
    """Test cases for VectorStore class."""

    def setup_method(self):
        """Setup test environment."""
        self.vector_store = VectorStore(max_size=5)

    def test_add_item(self):
        """Test adding items to vector store."""
        embedding = [1.0, 2.0, 3.0]
        document = "Test document"
        metadata = {"source": "test"}

        self.vector_store.add_item(embedding, document, metadata)

        assert self.vector_store.size() == 1

    def test_search_empty(self):
        """Test searching empty vector store."""
        query = [1.0, 2.0, 3.0]
        results = self.vector_store.search(query)

        assert results == []

    def test_search_with_items(self):
        """Test searching with items in vector store."""
        # Add items
        self.vector_store.add_item([1.0, 0.0, 0.0], "Document 1")
        self.vector_store.add_item([0.0, 1.0, 0.0], "Document 2")
        self.vector_store.add_item([0.0, 0.0, 1.0], "Document 3")

        # Search for first document
        results = self.vector_store.search([1.0, 0.0, 0.0], top_k=1)

        assert len(results) == 1
        assert results[0][1] == "Document 1"  # Check document content
        assert results[0][0] > 0.9  # Check similarity

    def test_max_size_limit(self):
        """Test that vector store respects max size limit."""
        # Add items beyond max_size
        for i in range(7):
            self.vector_store.add_item([i, i + 1], f"Document {i}")

        assert self.vector_store.size() == 5  # Should be limited to max_size

    def test_clear(self):
        """Test clearing vector store."""
        self.vector_store.add_item([1.0, 2.0], "Test document")
        assert self.vector_store.size() == 1

        self.vector_store.clear()
        assert self.vector_store.size() == 0

    def test_get_stats(self):
        """Test getting vector store statistics."""
        stats = self.vector_store.get_stats()

        assert "size" in stats
        assert "embedding_dim" in stats
        assert "max_size" in stats
        assert stats["size"] == 0

        # Add an item and check stats
        self.vector_store.add_item([1.0, 2.0, 3.0], "Test")
        stats = self.vector_store.get_stats()

        assert stats["size"] == 1
        assert stats["embedding_dim"] == 3