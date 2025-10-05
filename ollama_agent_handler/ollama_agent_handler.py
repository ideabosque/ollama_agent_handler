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

import ollama
import pendulum

from ai_agent_handler import AIAgentEventHandler
from silvaengine_utility import Utility


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

        self.system_message = {"role": "system", "content": agent["instructions"]}
        self.model_setting = {
            k: float(v) if isinstance(v, Decimal) else v
            for k, v in agent["configuration"].items()
        }
        self.client = ollama.Client(
            host=self.model_setting.get("base_url"),
            headers=self.model_setting.get("headers", {}),
        )

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
                messages.append(input_message)

            # Prepare chat parameters
            chat_params = {
                "model": self.model_setting.get("model"),
                "messages": messages,
            }

            # Add optional parameters in the 'options' dict (Ollama format)
            options = {}
            if "temperature" in self.model_setting:
                options["temperature"] = self.model_setting["temperature"]
            if "top_p" in self.model_setting:
                options["top_p"] = self.model_setting["top_p"]
            if "num_predict" in self.model_setting:
                options["num_predict"] = self.model_setting["num_predict"]

            if options:
                chat_params["options"] = options

            # Add tools if available
            if "tools" in self.model_setting:
                chat_params["tools"] = self.model_setting["tools"]

            # Add format option if specified
            text_config = self.model_setting.get("text", {})
            format_config = text_config.get("format", {})
            format_type = format_config.get("type", "text")

            if format_type == "json_object":
                chat_params["format"] = "json"
            elif format_type == "json_schema":
                # Ollama supports json schema format
                if "schema" in format_config:
                    chat_params["format"] = format_config["schema"]

            # Return streaming or non-streaming response
            if kwargs["stream"]:
                chat_params["stream"] = True

            return self.client.chat(**chat_params)

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
                self.handle_stream(response, input_messages, stream_event=stream_event)
                return None

            self.handle_response(response, input_messages)
            return run_id
        except Exception as e:
            self.logger.error(f"Error in ask_model: {str(e)}")
            raise Exception(f"Failed to process model request: {str(e)}")

    def handle_function_call(
        self, tool_call: Dict[str, any], input_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Processes and executes tool/function calls from model responses

        Args:
            tool_call: Tool call data from model response (Ollama format)

        Returns:
            Dict containing function execution results in Ollama's tool message format

        Raises:
            ValueError: For invalid tool calls
            Exception: For function execution failures
        """
        # Ollama format: {"function": {"name": "...", "arguments": {...}}}
        if "function" not in tool_call:
            raise ValueError("Invalid tool_call object: missing 'function' key")

        function_info = tool_call["function"]
        function_call_data = {
            "id": str(uuid.uuid4()),  # Generate ID if not provided
            "arguments": function_info.get("arguments", {}),
            "type": "function",
            "name": function_info["name"],
        }

        try:
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
            content = Utility.json_dumps(
                {
                    "tool": {
                        "tool_call_id": function_call_data["id"],
                        "tool_type": function_call_data["type"],
                        "name": function_call_data["name"],
                        "arguments": arguments,
                    },
                    "output": function_output,
                }
            )

            if self._run is None:
                self._short_term_memory.append(
                    {
                        "message": {
                            "role": self.agent["tool_call_role"],
                            "content": content,
                        },
                        "created_at": pendulum.now("UTC"),
                    }
                )

            # Return in Ollama's tool message format
            # Append tool result in Ollama format with "tool" role
            # Ensure content is a JSON string
            content = (
                Utility.json_dumps(function_output)
                if not isinstance(function_output, str)
                else function_output
            )
            input_messages.append(
                {
                    "role": "tool",
                    "content": content,
                    "tool_name": function_call_data["name"],
                }
            )
            return input_messages

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
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Processes non-streaming model output

        Args:
            response: Model response object (Ollama format)
        """
        self.logger.info("Processing output: %s", response)

        # Ollama response format: {"message": {"role": "assistant", "content": "..."}, "model": "...", ...}
        message = response.get("message", {})

        # Check for tool calls in the response
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = message["tool_calls"]

            # First, append the assistant's message with tool_calls
            input_messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content", ""),
                    "tool_calls": tool_calls,
                }
            )

            # Then, append tool results
            for tool_call in tool_calls:
                input_messages = self.handle_function_call(tool_call, input_messages)

            # Make follow-up call with tool results
            response = self.invoke_model(
                **{"input_messages": input_messages, "stream": False}
            )
            self.handle_response(response, input_messages)
            return

        # Generate a unique message ID since Ollama doesn't provide one
        message_id = f"msg-{pendulum.now('UTC').int_timestamp}-{str(uuid.uuid4())[:8]}"
        self.final_output = {
            "message_id": message_id,
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
        }

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]],
        stream_event: threading.Event = None,
    ) -> None:
        """
        Processes streaming model responses chunk by chunk

        Args:
            response_stream: Iterator of response chunks (Ollama format)
            stream_event: Event to signal completion

        Handles:
            - Accumulating response text
            - Processing JSON vs text formats
            - Handling tool calls in streaming chunks
            - Sending chunks to websocket
            - Signaling completion
        """
        message_id = None
        self.accumulated_text = ""
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        accumulated_tool_calls = []
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
            # Ollama streaming format: {"message": {"role": "assistant", "content": "..."}, "done": false, ...}
            # Generate message_id on first chunk
            if not message_id:
                message_id = (
                    f"msg-{pendulum.now('UTC').int_timestamp}-{str(uuid.uuid4())[:8]}"
                )

            # Get the message from the chunk
            message = chunk.get("message", {})

            # Check for tool calls in streaming chunks (Ollama v0.8.0+)
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    # Accumulate tool calls to handle them at the end
                    accumulated_tool_calls.append(tool_call)

            # Get the content from the chunk
            chunk_content = message.get("content", "")

            # Skip empty chunks
            if not chunk_content:
                continue

            # Print out for stream.
            print(chunk_content, end="", flush=True)
            if output_format in ["json_object", "json_schema"]:
                accumulated_partial_json += chunk_content
                index, self.accumulated_text, accumulated_partial_json = (
                    self.process_and_send_json(
                        index,
                        self.accumulated_text,
                        accumulated_partial_json,
                        output_format,
                    )
                )
            else:
                self.accumulated_text += chunk_content
                accumulated_partial_text += chunk_content
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

        # Handle accumulated tool calls after streaming completes
        if accumulated_tool_calls:
            # First, append the assistant's message with tool_calls
            input_messages.append(
                {
                    "role": "assistant",
                    "content": self.accumulated_text,
                    "tool_calls": accumulated_tool_calls,
                }
            )

            # Then, append tool results
            for tool_call in accumulated_tool_calls:
                input_messages = self.handle_function_call(tool_call, input_messages)

            # Make follow-up streaming call with tool results
            # Note: Recursive call will handle its own stream_event and final_output
            response = self.invoke_model(
                **{"input_messages": input_messages, "stream": True}
            )
            self.handle_stream(response, input_messages, stream_event=stream_event)
            return

        # Send final message end signal
        self.send_data_to_stream(
            index=index,
            data_format=output_format,
            is_message_end=True,
        )

        # Set final output
        self.final_output = {
            "message_id": (
                message_id
                if message_id
                else f"msg-{pendulum.now('UTC').int_timestamp}-{str(uuid.uuid4())[:8]}"
            ),
            "role": "assistant",
            "content": self.accumulated_text,
        }

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()
