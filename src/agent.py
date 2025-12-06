"""
Agent module for orchestrating LLM, MCP, and RAG interactions.
Provides the main interface for the intelligent agent system.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
import uuid
from .llm_client import LLMClient, Tool
from .mcp_client import MCPManager, MCPTool
from .embedding_retriever import EmbeddingRetriever
from .config import Config


class Agent:
    """
    Main agent class that orchestrates LLM, MCP, and RAG interactions.
    Handles tool calling, context management, and conversation flow.
    """

    def __init__(
        self,
        model: str = None,
        mcp_configs: List[Dict[str, Any]] = None,
        system_prompt: str = "",
        rag_context: str = ""
    ):
        """
        Initialize the agent.

        Args:
            model: LLM model to use
            mcp_configs: List of MCP client configurations
            system_prompt: System prompt for the agent
            rag_context: RAG context to include
        """
        self.model = model or Config.OPENAI_MODEL
        self.mcp_configs = mcp_configs or []
        self.system_prompt = self._build_system_prompt(system_prompt, rag_context)

        # Initialize components
        self.llm_client = LLMClient(
            model=self.model,
            system_prompt=self.system_prompt
        )

        self.mcp_manager = MCPManager()
        self.embedding_retriever: Optional[EmbeddingRetriever] = None

        # Track initialization state
        self.initialized = False

    def _build_system_prompt(self, system_prompt: str, rag_context: str) -> str:
        """
        Build the complete system prompt with RAG context.

        Args:
            system_prompt: Base system prompt
            rag_context: RAG context information

        Returns:
            Complete system prompt
        """
        base_prompt = system_prompt or "You are a helpful AI assistant."

        if rag_context:
            context_section = f"""
CONTEXT INFORMATION:
{rag_context}

Please use the above context information to help answer questions. Be specific about what information comes from the provided context.
"""
            return f"{base_prompt}\n{context_section}"

        return base_prompt

    async def init(self, knowledge_base_path: str = None) -> None:
        """
        Initialize the agent components.

        Args:
            knowledge_base_path: Path to knowledge base for RAG
        """
        try:
            # Initialize MCP clients
            for config in self.mcp_configs:
                self.mcp_manager.add_client(
                    name=config["name"],
                    command=config["command"],
                    args=config["args"]
                )

            await self.mcp_manager.init_all()

            # Setup tools from MCP clients
            await self._setup_tools()

            # Initialize RAG if knowledge base path is provided
            if knowledge_base_path:
                await self._init_rag(knowledge_base_path)

            self.initialized = True
            print("Agent initialized successfully")

        except Exception as e:
            print(f"Failed to initialize agent: {e}")
            await self.close()
            raise

    async def _setup_tools(self) -> None:
        """Setup tools from MCP clients for the LLM."""
        tools = []
        all_tools = self.mcp_manager.get_all_tools()

        for client_name, mcp_tools in all_tools.items():
            for mcp_tool in mcp_tools:
                tool = Tool(
                    name=f"{client_name}_{mcp_tool.name}",
                    description=f"[{client_name}] {mcp_tool.description}",
                    parameters=mcp_tool.input_schema
                )
                tools.append(tool)

        self.llm_client.set_tools(tools)
        print(f"Setup {len(tools)} tools from MCP clients")

    async def _init_rag(self, knowledge_base_path: str) -> None:
        """
        Initialize RAG with knowledge base.

        Args:
            knowledge_base_path: Path to knowledge base
        """
        self.embedding_retriever = EmbeddingRetriever()
        await self.embedding_retriever.index_directory(knowledge_base_path)

        stats = self.embedding_retriever.get_stats()
        print(f"RAG initialized with {stats['size']} documents")

    async def invoke(self, prompt: str, stream: bool = True) -> str:
        """
        Invoke the agent with a prompt.

        Args:
            prompt: User prompt
            stream: Whether to stream the response

        Returns:
            Agent response
        """
        if not self.initialized:
            raise RuntimeError("Agent not initialized")

        # Add RAG context if available
        if self.embedding_retriever:
            try:
                rag_context = await self.embedding_retriever.retrieve_context(prompt)
                if rag_context and rag_context != "No relevant context found.":
                    enhanced_prompt = f"""
USER QUERY: {prompt}

RELEVANT CONTEXT:
{rag_context}

Please answer the user's query using the relevant context provided above.
"""
                    prompt = enhanced_prompt
            except Exception as e:
                print(f"Error retrieving RAG context: {e}")

        if stream:
            return await self._invoke_with_streaming(prompt)
        else:
            return await self._invoke_without_streaming(prompt)

    async def _invoke_with_streaming(self, prompt: str) -> str:
        """
        Invoke agent with streaming and tool calling.

        Args:
            prompt: User prompt

        Returns:
            Complete response
        """
        response_chunks = []
        max_tool_iterations = 10
        tool_iteration = 0

        while tool_iteration < max_tool_iterations:
            # Stream LLM response
            async for chunk in self.llm_client.chat_stream(prompt):
                response_chunks.append(chunk)
                print(chunk, end='', flush=True)  # Real-time output

            print()  # New line after streaming

            # Check for tool calls
            tool_calls = self.llm_client.get_tool_calls()

            if not tool_calls:
                break  # No more tool calls, return response

            # Execute tools
            for tool_call in tool_calls:
                try:
                    result = await self._execute_tool(tool_call)
                    self.llm_client.append_tool_result(tool_call.id, str(result))

                    # Add tool result to response for context
                    tool_response = f"\n[Tool: {tool_call.function_name}]\n{result}\n"
                    response_chunks.append(tool_response)
                    print(tool_response)

                except Exception as e:
                    error_msg = f"Error executing tool {tool_call.function_name}: {e}"
                    self.llm_client.append_tool_result(tool_call.id, error_msg)
                    print(f"Error: {error_msg}")

            tool_iteration += 1

        return "".join(response_chunks)

    async def _invoke_without_streaming(self, prompt: str) -> str:
        """
        Invoke agent without streaming.

        Args:
            prompt: User prompt

        Returns:
            Complete response
        """
        max_tool_iterations = 10
        tool_iteration = 0
        full_response = ""

        while tool_iteration < max_tool_iterations:
            # Get LLM response
            response = await self.llm_client.chat(prompt)
            full_response += response

            # Check for tool calls
            tool_calls = self.llm_client.get_tool_calls()

            if not tool_calls:
                break  # No more tool calls

            # Execute tools
            for tool_call in tool_calls:
                try:
                    result = await self._execute_tool(tool_call)
                    self.llm_client.append_tool_result(tool_call.id, str(result))
                    full_response += f"\n[Tool: {tool_call.function_name}]\n{result}\n"

                except Exception as e:
                    error_msg = f"Error executing tool {tool_call.function_name}: {e}"
                    self.llm_client.append_tool_result(tool_call.id, error_msg)
                    full_response += f"\nError: {error_msg}\n"

            tool_iteration += 1

        return full_response

    async def _execute_tool(self, tool_call) -> str:
        """
        Execute a tool call.

        Args:
            tool_call: Tool call object

        Returns:
            Tool execution result
        """
        tool_name = tool_call.function_name
        tool_arguments = tool_call.arguments

        # Parse client name and tool name
        if "_" in tool_name:
            client_name, actual_tool_name = tool_name.split("_", 1)
        else:
            # Fallback: try all clients
            client_name = None
            actual_tool_name = tool_name

        # Try to execute the tool
        if client_name:
            try:
                result = await self.mcp_manager.call_tool(
                    client_name=client_name,
                    tool_name=actual_tool_name,
                    arguments=tool_arguments
                )
                return json.dumps(result, indent=2, ensure_ascii=False)
            except Exception as e:
                # Try other clients as fallback
                for client in self.mcp_manager.clients.values():
                    if client.initialized and actual_tool_name in client.tools:
                        try:
                            result = await client.call_tool(actual_tool_name, tool_arguments)
                            return json.dumps(result, indent=2, ensure_ascii=False)
                        except Exception:
                            continue
                raise e
        else:
            # Try all clients
            for client_name, client in self.mcp_manager.clients.items():
                if client.initialized and actual_tool_name in client.tools:
                    try:
                        result = await client.call_tool(actual_tool_name, tool_arguments)
                        return json.dumps(result, indent=2, ensure_ascii=False)
                    except Exception:
                        continue

            raise ValueError(f"Tool '{tool_name}' not found in any MCP client")

    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get agent capabilities and statistics.

        Returns:
            Dictionary with capabilities information
        """
        capabilities = {
            "model": self.model,
            "initialized": self.initialized,
            "mcp_clients": {},
            "tools_count": 0,
            "rag_enabled": self.embedding_retriever is not None
        }

        # MCP client information
        all_tools = self.mcp_manager.get_all_tools()
        for client_name, tools in all_tools.items():
            capabilities["mcp_clients"][client_name] = {
                "tools_count": len(tools),
                "tools": [tool.name for tool in tools]
            }
            capabilities["tools_count"] += len(tools)

        # RAG information
        if self.embedding_retriever:
            rag_stats = self.embedding_retriever.get_stats()
            capabilities["rag_stats"] = rag_stats

        return capabilities

    def clear_conversation(self) -> None:
        """Clear conversation history while keeping system prompt."""
        self.llm_client.clear_history()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.

        Returns:
            List of messages in the conversation
        """
        return self.llm_client.get_conversation_history()

    async def close(self) -> None:
        """Close the agent and cleanup resources."""
        # Close MCP clients
        await self.mcp_manager.close_all()

        # Close LLM client
        await self.llm_client.close()

        # Close embedding retriever
        if self.embedding_retriever:
            await self.embedding_retriever.close()

        self.initialized = False
        print("Agent closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()