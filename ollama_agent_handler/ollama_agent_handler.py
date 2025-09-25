#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

__author__ = "bibow"

import logging
import threading
import traceback
import uuid
from decimal import Decimal
from queue import Queue
from typing import Any, Dict, List, Optional

import pendulum
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from ai_agent_handler import AIAgentEventHandler
from silvaengine_utility import Utility


class PrintStreamingCallback(BaseCallbackHandler):
    """Callback handler that prints streaming tokens to stdout"""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)


# ----------------------------
# Ollama Response Streaming with Function Handling and History
# ----------------------------
class OllamaEventHandler(AIAgentEventHandler):
    """
    Manages conversations and function calls in real-time with Ollama:
      - Handles streaming responses from the model
      - Processes tool/function calls in responses
      - Executes functions and manages their lifecycle
      - Maintains conversation context and history
      - Handles both streaming and non-streaming responses
    """

    def __init__(
        self,
        logger: logging.Logger,
        agent: Dict[str, Any],
        **setting: Dict[str, Any],
    ) -> None:
        """
        Initialize the Ollama event handler

        Args:
            logger: Logger instance for debug/info messages
            agent: Configuration dict containing model instructions and settings
            setting: Additional settings for handler configuration
        """
        AIAgentEventHandler.__init__(self, logger, agent, **setting)

        self.system_message = SystemMessage(content=agent["instructions"])
        self.model_setting = {
            k: float(v) if isinstance(v, Decimal) else v
            for k, v in agent["configuration"].items()
            if k not in ["tools", "text"]
        }

        self.tools = []
        for tool in agent["configuration"].get("tools", []):
            self.tools.append(self.get_class(tool["module_name"], tool["class_name"]))

    def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Invokes the Ollama model with provided messages and handles tool calls

        Args:
            kwargs: Contains input messages and streaming configuration

        Returns:
            Model response or streaming response

        Raises:
            Exception: If model invocation fails
        """
        try:
            messages = []
            messages.append(self.system_message)
            for input_message in kwargs.get("input_messages"):
                if input_message["role"] == "user":
                    messages.append(HumanMessage(content=input_message["content"]))
                elif input_message["role"] == "assistant":
                    messages.append(AIMessage(content=input_message["content"]))
                elif input_message["role"] == self.agent["tool_call_role"]:
                    tool_call_id = Utility.json_loads(input_message["content"])["tool"][
                        "tool_call_id"
                    ]
                    messages.append(
                        ToolMessage(
                            tool_call_id=tool_call_id, content=input_message["content"]
                        )
                    )

            model = ChatOllama(**self.model_setting)
            model_with_tools: Runnable = model.bind_tools(self.tools)

            response = model_with_tools.invoke(messages)
            if hasattr(response, "tool_calls") and len(response.tool_calls) > 0:
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_message = self.handle_function_call(tool_call)
                    tool_messages.append(tool_message)

                # Follow-up round
                messages = (
                    messages
                    + [
                        AIMessage(
                            content=response.content, tool_calls=response.tool_calls
                        )
                    ]
                    + tool_messages
                )

            if not kwargs["stream"]:
                return model.invoke(messages)

            callback = PrintStreamingCallback()
            return model.stream(messages, config={"callbacks": [callback]})
        except Exception as e:
            self.logger.error(f"Error invoking model: {str(e)}")
            raise Exception(f"Failed to invoke model: {str(e)}")

    @Utility.performance_monitor.monitor_operation(operation_name="Ollama")
    def ask_model(
        self,
        input_messages: List[Dict[str, Any]],
        queue: Queue = None,
        stream_event: threading.Event = None,
        model_setting: Dict[str, Any] = None,
    ) -> Optional[str]:
        """
        Makes a request to the Ollama model with streaming or non-streaming mode

        Args:
            input_messages: List of conversation messages
            queue: Queue for streaming events
            stream_event: Event to signal streaming completion
            model_setting: Optional model configuration overrides

        Returns:
            Response ID for non-streaming requests, None for streaming

        Raises:
            Exception: If request processing fails
        """
        try:
            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                self.model_setting.update(model_setting)

            timestamp = pendulum.now("UTC").int_timestamp
            run_id = (
                f"run-{self.model_setting["model"]}-{timestamp}-{str(uuid.uuid4())[:8]}"
            )

            response = self.invoke_model(
                **{
                    "input_messages": input_messages,
                    "stream": stream,
                }
            )

            if stream:
                queue.put({"name": "run_id", "value": run_id})
                self.handle_stream(response, stream_event=stream_event)
                return None

            self.handle_response(response)
            return run_id
        except Exception as e:
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")

    def handle_function_call(self, tool_call: Dict[str, any]) -> ToolMessage:
        """
        Processes and executes tool/function calls from model responses

        Args:
            tool_call: Tool call data from model response

        Returns:
            ToolMessage containing function execution results

        Raises:
            ValueError: For invalid tool calls
            Exception: For function execution failures
        """
        if not "id" in tool_call:
            raise ValueError("Invalid tool_call object")

        try:
            function_call_data = {
                "id": tool_call["id"],
                "arguments": tool_call["args"],
                "type": tool_call["type"],
                "name": tool_call["name"],
            }

            self.logger.info(
                f"[handle_function_call] Starting function call recording for {function_call_data['name']}"
            )
            self._record_function_call_start(function_call_data)

            self.logger.info(
                f"[handle_function_call] Processing arguments for function {function_call_data['name']}"
            )
            arguments = self._process_function_arguments(function_call_data)

            self.logger.info(
                f"[handle_function_call] Executing function {function_call_data['name']} with arguments {arguments}"
            )
            function_output = self._execute_function(function_call_data, arguments)

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": self.agent["tool_call_role"],
                            "content": Utility.json_dumps(
                                {
                                    "tool": {
                                        "tool_call_id": function_call_data["id"],
                                        "tool_type": function_call_data["type"],
                                        "name": function_call_data["name"],
                                        "arguments": arguments,
                                    },
                                    "output": function_output,
                                }
                            ),
                        },
                        "created_at": pendulum.now("UTC"),
                    }
                )

            return ToolMessage(
                tool_call_id=function_call_data["id"], content=str(function_output)
            )

        except Exception as e:
            self.logger.error(f"Error in handle_function_call: {e}")
            raise

    def _record_function_call_start(self, function_call_data: Dict[str, Any]) -> None:
        """
        Records initial function call metadata to storage

        Args:
            function_call_data: Function call details to record
        """
        self.invoke_async_funct(
            "async_insert_update_tool_call",
            **{
                "tool_call_id": function_call_data["id"],
                "tool_type": function_call_data["type"],
                "name": function_call_data["name"],
            },
        )

    def _process_function_arguments(
        self, function_call_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parses and validates function arguments from tool call

        Args:
            function_call_data: Raw function call data

        Returns:
            Processed arguments dictionary

        Raises:
            ValueError: If argument parsing fails
        """
        try:
            arguments = function_call_data.get("arguments", {})
            arguments["endpoint_id"] = self._endpoint_id

            return arguments

        except Exception as e:
            log = traceback.format_exc()
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": function_call_data.get("arguments", "{}"),
                    "status": "failed",
                    "notes": log,
                },
            )
            self.logger.error("Error parsing function arguments: %s", e)
            raise ValueError(f"Failed to parse function arguments: {e}")

    def _execute_function(
        self, function_call_data: Dict[str, Any], arguments: Dict[str, Any]
    ) -> Any:
        """
        Executes the requested function and handles results/errors

        Args:
            function_call_data: Function metadata
            arguments: Processed function arguments

        Returns:
            Function execution output

        Raises:
            ValueError: For unsupported functions
        """
        agent_function = self.get_function(function_call_data["name"])
        if not agent_function:
            raise ValueError(
                f"Unsupported function requested: {function_call_data['name']}"
            )

        try:
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": Utility.json_dumps(arguments),
                    "status": "in_progress",
                },
            )

            function_output = agent_function(**arguments)

            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "content": Utility.json_dumps(function_output),
                    "status": "completed",
                },
            )
            return function_output

        except Exception as e:
            log = traceback.format_exc()
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": Utility.json_dumps(arguments),
                    "status": "failed",
                    "notes": log,
                },
            )
            return f"Function execution failed: {e}"

    def handle_response(
        self,
        response: Any,
    ) -> None:
        """
        Processes non-streaming model output

        Args:
            response: Model response object
        """
        self.logger.info("Processing output: %s", response)

        self.final_output = {
            "message_id": response.id,
            "role": "assistant",
            "content": response.content,
        }

    def handle_stream(
        self,
        response_stream: Any,
        stream_event: threading.Event = None,
    ) -> None:
        """
        Processes streaming model responses chunk by chunk

        Args:
            response_stream: Iterator of response chunks
            stream_event: Event to signal completion

        Handles:
            - Accumulating response text
            - Processing JSON vs text formats
            - Sending chunks to websocket
            - Signaling completion
        """
        message_id = None
        self.accumulated_text = ""
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        output_format = (
            self.model_setting.get("text", {"format": {"type": "text"}})
            .get("format", {"type": "text"})
            .get("type", "text")
        )
        index = 0

        self.send_data_to_stream(
            index=index,
            data_format=output_format,
        )
        index += 1

        for chunk in response_stream:
            if not message_id:
                message_id = chunk.id
            if not isinstance(chunk, AIMessageChunk):
                continue

            if output_format in ["json_object", "json_schema"]:
                accumulated_partial_json += chunk.content
                index, self.accumulated_text, accumulated_partial_json = (
                    self.process_and_send_json(
                        index,
                        self.accumulated_text,
                        accumulated_partial_json,
                        output_format,
                    )
                )
            else:
                self.accumulated_text += chunk.content
                accumulated_partial_text += chunk.content
                # Check if text contains XML-style tags and update format
                index, accumulated_partial_text = self.process_text_content(
                    index, accumulated_partial_text, output_format
                )

        if len(accumulated_partial_text) > 0:
            self.send_data_to_stream(
                index=index,
                data_format=output_format,
                chunk_delta=accumulated_partial_text,
            )
            accumulated_partial_text = ""
            index += 1

        self.send_data_to_stream(
            index=index,
            data_format=output_format,
            is_message_end=True,
        )

        self.final_output = {
            "message_id": message_id,
            "role": "assistant",
            "content": self.accumulated_text,
        }

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()
