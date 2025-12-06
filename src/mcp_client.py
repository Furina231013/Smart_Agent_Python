"""
MCP (Model Context Protocol) Client module.
Handles communication with MCP servers via stdio transport.
Supports tool discovery, calling, and connection management.
"""

import json
import asyncio
import subprocess
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid
from .config import Config


@dataclass
class MCPTool:
    """Represents a tool from an MCP server."""
    name: str
    description: str
    input_schema: Dict[str, Any]


class MCPClient:
    """
    Client for communicating with MCP servers via stdio transport.
    Supports tool discovery, calling, and connection lifecycle management.
    """

    def __init__(self, name: str, command: str, args: List[str]):
        """
        Initialize the MCP client.

        Args:
            name: Name identifier for this MCP client
            command: Command to start the MCP server
            args: Arguments for the command
        """
        self.name = name
        self.command = command
        self.args = args
        self.process: Optional[subprocess.Popen] = None
        self.transport = None
        self.session = None
        self.tools: Dict[str, MCPTool] = {}
        self.initialized = False
        self.message_id = 1

    async def init(self) -> None:
        """Initialize connection to MCP server and discover tools."""
        try:
            # Start the MCP server process
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            if not self.process.stdin or not self.process.stdout:
                raise RuntimeError("Failed to create stdio transport")

            # Initialize the MCP session
            await self._initialize_session()

            # Discover available tools
            await self._discover_tools()

            self.initialized = True
            print(f"MCP client '{self.name}' initialized successfully")

        except Exception as e:
            print(f"Failed to initialize MCP client '{self.name}': {e}")
            await self.close()
            raise

    async def _initialize_session(self) -> None:
        """Initialize the MCP session with the server."""
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "python-mcp-client",
                    "version": "1.0.0"
                }
            }
        }

        response = await self._send_request(init_request)
        if "error" in response:
            raise RuntimeError(f"Initialize failed: {response['error']}")

        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }

        await self._send_notification(initialized_notification)

    async def _discover_tools(self) -> None:
        """Discover available tools from the MCP server."""
        tools_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }

        response = await self._send_request(tools_request)
        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")

        tools = response.get("result", {}).get("tools", [])
        for tool in tools:
            mcp_tool = MCPTool(
                name=tool["name"],
                description=tool.get("description", ""),
                input_schema=tool.get("inputSchema", {})
            )
            self.tools[tool["name"]] = mcp_tool

        print(f"Discovered {len(self.tools)} tools from MCP client '{self.name}'")

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            name: Name of the tool to call
            arguments: Arguments for the tool

        Returns:
            Tool execution result

        Raises:
            RuntimeError: If the client is not initialized or tool call fails
        """
        if not self.initialized:
            raise RuntimeError("MCP client not initialized")

        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")

        call_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }

        response = await self._send_request(call_request)
        if "error" in response:
            raise RuntimeError(f"Tool call failed: {response['error']}")

        return response.get("result", {})

    def get_tools(self) -> List[MCPTool]:
        """
        Get all available tools.

        Returns:
            List of available tools
        """
        return list(self.tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """
        Get a specific tool by name.

        Args:
            name: Name of the tool

        Returns:
            Tool if found, None otherwise
        """
        return self.tools.get(name)

    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and wait for response.

        Args:
            request: JSON-RPC request

        Returns:
            JSON-RPC response
        """
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Process not initialized")

        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()

        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from MCP server")

        try:
            response = json.loads(response_line.decode().strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}")

        return response

    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """
        Send a JSON-RPC notification (no response expected).

        Args:
            notification: JSON-RPC notification
        """
        if not self.process or not self.process.stdin:
            raise RuntimeError("Process not initialized")

        notification_json = json.dumps(notification) + "\n"
        self.process.stdin.write(notification_json.encode())
        await self.process.stdin.drain()

    def _next_id(self) -> int:
        """Get the next message ID."""
        current_id = self.message_id
        self.message_id += 1
        return current_id

    async def close(self) -> None:
        """Close the MCP client connection and cleanup resources."""
        if self.initialized:
            # Send close notification if session is active
            try:
                close_notification = {
                    "jsonrpc": "2.0",
                    "method": "close"
                }
                await self._send_notification(close_notification)
            except Exception as e:
                print(f"Error sending close notification: {e}")

        # Terminate the process
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except Exception as e:
                print(f"Error terminating process: {e}")
            finally:
                self.process = None

        self.initialized = False
        self.tools.clear()
        print(f"MCP client '{self.name}' closed")

    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.process and self.process.poll() is None:
            self.process.terminate()


class MCPManager:
    """
    Manager for multiple MCP clients.
    Provides a unified interface for managing multiple MCP connections.
    """

    def __init__(self):
        """Initialize the MCP manager."""
        self.clients: Dict[str, MCPClient] = {}

    def add_client(self, name: str, command: str, args: List[str]) -> None:
        """
        Add an MCP client to the manager.

        Args:
            name: Name identifier for the client
            command: Command to start the MCP server
            args: Arguments for the command
        """
        client = MCPClient(name, command, args)
        self.clients[name] = client

    async def init_all(self) -> None:
        """Initialize all MCP clients."""
        init_tasks = []
        for name, client in self.clients.items():
            init_tasks.append(self._init_client_safely(name, client))

        if init_tasks:
            await asyncio.gather(*init_tasks, return_exceptions=True)

    async def _init_client_safely(self, name: str, client: MCPClient) -> None:
        """Initialize a single client with error handling."""
        try:
            await client.init()
        except Exception as e:
            print(f"Failed to initialize MCP client '{name}': {e}")

    def get_client(self, name: str) -> Optional[MCPClient]:
        """
        Get a specific MCP client by name.

        Args:
            name: Name of the client

        Returns:
            MCP client if found, None otherwise
        """
        return self.clients.get(name)

    def get_all_tools(self) -> Dict[str, List[MCPTool]]:
        """
        Get all tools from all clients.

        Returns:
            Dictionary mapping client names to their tools
        """
        all_tools = {}
        for name, client in self.clients.items():
            if client.initialized:
                all_tools[name] = client.get_tools()
        return all_tools

    async def call_tool(
        self,
        client_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call a tool on a specific client.

        Args:
            client_name: Name of the MCP client
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        client = self.get_client(client_name)
        if not client:
            raise ValueError(f"MCP client '{client_name}' not found")

        return await client.call_tool(tool_name, arguments)

    async def close_all(self) -> None:
        """Close all MCP clients."""
        close_tasks = []
        for name, client in self.clients.items():
            close_tasks.append(self._close_client_safely(name, client))

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self.clients.clear()

    async def _close_client_safely(self, name: str, client: MCPClient) -> None:
        """Close a single client with error handling."""
        try:
            await client.close()
        except Exception as e:
            print(f"Error closing MCP client '{name}': {e}")