# -*- coding: utf-8 -*-
"""
Main entry point for the LLM+MCP+RAG Python framework.
Demonstrates usage of the agent system with RAG context.
"""

import asyncio
import os
from .agent import Agent
from .config import Config


async def retrieve_context(prompt: str, knowledge_base_path: str = None):
    """
    Retrieve relevant context for a prompt.

    Args:
        prompt: Query prompt
        knowledge_base_path: Path to knowledge base

    Returns:
        Retrieved context
    """
    from .embedding_retriever import EmbeddingRetriever

    if not knowledge_base_path or not os.path.exists(knowledge_base_path):
        return ""

    retriever = EmbeddingRetriever()
    try:
        await retriever.index_directory(knowledge_base_path)
        context = await retriever.retrieve_context(prompt)
        return context
    finally:
        await retriever.close()


async def main():
    """Main function demonstrating the agent usage."""
    # Validate configuration
    Config.validate()

    # Example usage based on the TypeScript version
    prompt = "告诉我Antonette的信息,先从我给你的上下文中找到相关信息,总结后创作一个关于她的故事,形成一段文字描述,要求包含她的基本信息和故事"

    print("Python LLM+MCP+RAG Framework")
    print("=" * 50)

    try:
        # Retrieve context if knowledge base exists
        knowledge_base_path = Config.KNOWLEDGE_BASE_PATH
        context = ""
        if os.path.exists(knowledge_base_path):
            print("Loading knowledge base...")
            context = await retrieve_context(prompt, knowledge_base_path)
            print(f"Context loaded: {len(context)} characters")

        # Create agent with RAG context
        agent = Agent(
            model="GLM-4-Flash-250414",  # or any other model
            mcp_configs=[],  # Add MCP configurations here if needed
            system_prompt="你是一个专业的AI助手，能够基于提供的上下文信息回答问题，并进行创作。请确保回答准确、有趣且富有想象力。",
            rag_context=context
        )

        # Initialize agent
        await agent.init()

        # Get agent capabilities
        capabilities = await agent.get_capabilities()
        print(f"\nAgent Capabilities:")
        print(f"- Model: {capabilities['model']}")
        print(f"- Tools available: {capabilities['tools_count']}")
        print(f"- RAG enabled: {capabilities['rag_enabled']}")

        if capabilities.get('rag_stats'):
            print(f"- Documents indexed: {capabilities['rag_stats']['size']}")

        # Invoke agent with streaming
        print(f"\nPrompt: {prompt}")
        print("\nResponse:")
        response = await agent.invoke(prompt, stream=True)

        print(f"\n\nComplete response length: {len(response)} characters")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())