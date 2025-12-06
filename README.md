# Python LLM+MCP+RAG Framework

A Python implementation of an intelligent agent framework that combines Large Language Models (LLM), Model Context Protocol (MCP), and Retrieval Augmented Generation (RAG).

## Features

- **LLM Integration**: Support for OpenAI-compatible APIs with streaming and function calling
- **MCP Support**: Integration with MCP servers for extensible tool usage
- **RAG System**: Document embedding and retrieval for enhanced context
- **Asynchronous Architecture**: Full async/await support for high performance
- **Vector Storage**: In-memory vector store with cosine similarity search
- **Configuration Management**: Environment-based configuration system

## Architecture

```
python_llm_mcp_rag/
├── src/
│   ├── __init__.py
│   ├── main.py              # Main entry point and demo
│   ├── agent.py             # Core agent orchestration
│   ├── llm_client.py        # LLM integration (OpenAI-compatible)
│   ├── mcp_client.py        # MCP protocol client
│   ├── embedding_retriever.py # RAG document retrieval
│   ├── vector_store.py      # Vector storage and search
│   └── config.py            # Configuration management
├── knowledge/users/         # Knowledge base directory
├── tests/                   # Test files
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
cd /Users/YourUserName/Documents/smart_agent/ts-node-esm-template/python_llm_mcp_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy the environment variables template:
```bash
cp .env.example .env
```

4. Edit `.env` with your API keys and configuration.

## Configuration

Required environment variables:

```bash
# OpenAI-compatible API
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Embedding service (e.g., SiliconFlow)
EMBEDDING_KEY=your_embedding_key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
```

Optional environment variables:

```bash
KNOWLEDGE_BASE_PATH=knowledge/users
VECTOR_STORE_SIZE=1000
SIMILARITY_THRESHOLD=0.7
MCP_TIMEOUT=30
```

## Usage

### Basic Usage

```python
import asyncio
from src.agent import Agent

async def main():
    # Create agent with RAG context
    agent = Agent(
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful AI assistant."
    )

    # Initialize with knowledge base
    await agent.init(knowledge_base_path="knowledge/users")

    # Get response with streaming
    response = await agent.invoke("Tell me about the users in the knowledge base.")
    print(response)

    # Cleanup
    await agent.close()

asyncio.run(main())
```

### Using MCP Tools

```python
# Configure MCP clients
mcp_configs = [
    {
        "name": "filesystem",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
    },
    {
        "name": "fetch",
        "command": "uvx",
        "args": ["mcp-server-fetch"]
    }
]

agent = Agent(
    model="gpt-3.5-turbo",
    mcp_configs=mcp_configs,
    system_prompt="You can use tools to help answer questions."
)
```

### Standalone RAG Usage

```python
from src.embedding_retriever import EmbeddingRetriever

async def search_documents(query: str):
    retriever = EmbeddingRetriever()

    # Index documents
    await retriever.index_directory("knowledge/users")

    # Retrieve relevant documents
    results = await retriever.retrieve(query, top_k=5)

    for similarity, document, metadata in results:
        print(f"Similarity: {similarity:.3f}")
        print(f"Source: {metadata.get('file_name', 'Unknown')}")
        print(f"Content: {document[:200]}...")
        print("-" * 50)
```

## Running the Demo

The included demo shows how to use the framework with RAG context:

```bash
python -m src.main
```

This will:
1. Load documents from the knowledge base
2. Create embeddings and index them
3. Process a sample query about "Antonette"
4. Generate a story using the retrieved context

## Key Components

### Agent (`src/agent.py`)
- Orchestrates LLM, MCP, and RAG interactions
- Handles tool calling and conversation management
- Provides unified interface for the agent system

### LLM Client (`src/llm_client.py`)
- OpenAI-compatible API client
- Streaming responses and function calling
- Conversation history management

### MCP Client (`src/mcp_client.py`)
- Model Context Protocol implementation
- Stdio transport for MCP server communication
- Tool discovery and execution

### Embedding Retriever (`src/embedding_retriever.py`)
- Document embedding and retrieval
- Integration with external embedding services
- Automatic document loading from directories

### Vector Store (`src/vector_store.py`)
- In-memory vector database
- Cosine similarity search
- Efficient vector operations with NumPy

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
```

### Type Checking

```bash
mypy src/
```

## Comparison with TypeScript Version

This Python implementation provides equivalent functionality to the TypeScript version:

| Feature | TypeScript | Python |
|---------|------------|---------|
| LLM Integration | ✅ OpenAI API | ✅ OpenAI API |
| MCP Support | ✅ Stdio Client | ✅ Stdio Client |
| RAG System | ✅ Embeddings + Vector Store | ✅ Embeddings + Vector Store |
| Streaming | ✅ Real-time responses | ✅ Real-time responses |
| Tool Calling | ✅ Function Calling | ✅ Function Calling |
| Async Support | ✅ Promises | ✅ Asyncio |

## License

This project is provided as-is for educational and development purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please create an issue in the repository.

<img width="2740" height="2402" alt="8fe810bbec860cea83973d6d17be79c8" src="https://github.com/user-attachments/assets/08192c22-99c2-40c4-8a29-773109fbaeb0" />
