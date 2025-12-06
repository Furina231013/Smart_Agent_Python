"""
Configuration module for the LLM+MCP+RAG framework.
Loads environment variables and provides configuration settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the framework."""

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    OPENAI_API_BASE_URL: str = os.getenv('OPENAI_API_BASE_URL', 'https://api.openai.com/v1')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

    # Embedding Configuration
    EMBEDDING_KEY: str = os.getenv('EMBEDDING_KEY', '')
    EMBEDDING_BASE_URL: str = os.getenv('EMBEDDING_BASE_URL', 'https://api.siliconflow.cn/v1')
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')

    # Knowledge Base Configuration
    KNOWLEDGE_BASE_PATH: str = os.getenv('KNOWLEDGE_BASE_PATH', 'knowledge/users')

    # Vector Store Configuration
    VECTOR_STORE_SIZE: int = int(os.getenv('VECTOR_STORE_SIZE', '1000'))
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))

    # MCP Configuration
    MCP_TIMEOUT: int = int(os.getenv('MCP_TIMEOUT', '30'))

    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")

        if not cls.EMBEDDING_KEY:
            raise ValueError("EMBEDDING_KEY is required")