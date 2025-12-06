"""
LLM Client module for interacting with OpenAI-compatible APIs.
Supports streaming responses, function calling, and tool management.
"""

import json
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator, Callable
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI
from .config import Config


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""
    id: str
    function_name: str
    arguments: Dict[str, Any]


@dataclass
class Tool:
    """Represents a tool definition for the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]


class LLMClient:
    """
    Client for interacting with OpenAI-compatible LLM APIs.
    Supports streaming, function calling, and conversation management.
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        system_prompt: str = ""
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model name to use
            api_key: OpenAI API key
            base_url: API base URL
            system_prompt: System prompt for the conversation
        """
        self.model = model or Config.OPENAI_MODEL
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = base_url or Config.OPENAI_API_BASE_URL
        self.system_prompt = system_prompt

        # Initialize async client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Conversation history
        self.messages: List[Dict[str, str]] = []

        # Available tools
        self.tools: List[Tool] = []

        # Add system prompt if provided
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def set_tools(self, tools: List[Tool]) -> None:
        """
        Set the available tools for the LLM.

        Args:
            tools: List of available tools
        """
        self.tools = tools

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions.

        Returns:
            List of tool definitions in OpenAI format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in self.tools
        ]

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Message role (user, assistant, system, tool)
            content: Message content
        """
        self.messages.append({"role": role, "content": content})

    def append_tool_result(self, tool_call_id: str, tool_result: str) -> None:
        """
        Append a tool execution result to the conversation.

        Args:
            tool_call_id: ID of the tool call
            tool_result: Result of the tool execution
        """
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": tool_result
        })

    async def chat_stream(
        self,
        prompt: Optional[str] = None,
        tools_enabled: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Send a message to the LLM and stream the response.

        Args:
            prompt: User prompt (if None, uses last user message)
            tools_enabled: Whether to enable tool calling

        Yields:
            Streamed response chunks
        """
        # Add user prompt if provided
        if prompt:
            self.add_message("user", prompt)

        # Prepare request parameters
        request_params = {
            "model": self.model,
            "messages": self.messages,
            "stream": True
        }

        # Add tools if available and enabled
        if tools_enabled and self.tools:
            request_params["tools"] = self.get_tool_definitions()

        try:
            # Create streaming chat completion
            response = await self.client.chat.completions.create(**request_params)

            accumulated_content = ""
            accumulated_tool_calls = []

            async for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta

                    # Handle content
                    if delta.content:
                        content = delta.content
                        accumulated_content += content
                        yield content

                    # Handle tool calls
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            # Find existing tool call or create new one
                            if len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls.append({
                                    "id": tool_call_delta.id or "",
                                    "function": {
                                        "name": "",
                                        "arguments": ""
                                    }
                                })

                            tool_call = accumulated_tool_calls[tool_call_delta.index]

                            if tool_call_delta.id:
                                tool_call["id"] = tool_call_delta.id

                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    tool_call["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

            # Add assistant response to history
            if accumulated_content or accumulated_tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": accumulated_content
                }

                if accumulated_tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"]
                            }
                        }
                        for tc in accumulated_tool_calls
                    ]

                self.messages.append(assistant_message)

        except Exception as e:
            yield f"Error: {str(e)}"

    async def chat(
        self,
        prompt: Optional[str] = None,
        tools_enabled: bool = True
    ) -> str:
        """
        Send a message to the LLM and get the complete response.

        Args:
            prompt: User prompt
            tools_enabled: Whether to enable tool calling

        Returns:
            Complete response from the LLM
        """
        response_chunks = []
        async for chunk in self.chat_stream(prompt, tools_enabled):
            response_chunks.append(chunk)

        return "".join(response_chunks)

    def get_tool_calls(self) -> List[ToolCall]:
        """
        Extract tool calls from the last assistant message.

        Returns:
            List of tool calls
        """
        tool_calls = []

        # Get the last assistant message
        for message in reversed(self.messages):
            if message.get("role") == "assistant":
                if "tool_calls" in message:
                    for tc in message["tool_calls"]:
                        try:
                            tool_call = ToolCall(
                                id=tc["id"],
                                function_name=tc["function"]["name"],
                                arguments=json.loads(tc["function"]["arguments"])
                            )
                            tool_calls.append(tool_call)
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"Error parsing tool call: {e}")
                break

        return tool_calls

    def clear_history(self) -> None:
        """Clear conversation history except system prompt."""
        system_messages = [msg for msg in self.messages if msg.get("role") == "system"]
        self.messages = system_messages

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.

        Returns:
            List of messages in the conversation
        """
        return self.messages.copy()

    async def close(self) -> None:
        """Close the client connection."""
        await self.client.close()