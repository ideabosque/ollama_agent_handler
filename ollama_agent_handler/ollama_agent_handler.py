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

        # Enable timeline logging (default: False)
        self.enable_timeline_log = setting.get("enable_timeline_log", False)

        self.system_message = {"role": "system", "content": agent["instructions"]}

        # Convert Decimal to float once during initialization (performance optimization)
        self.model_setting = {
            k: float(v) if isinstance(v, Decimal) else v
            for k, v in agent["configuration"].items()
        }

        # Cache frequently accessed configuration values (performance optimization)
        self.output_format_type = (
            self.model_setting.get("text", {}).get("format", {}).get("type", "text")
        )

        # Validate reasoning configuration if present
        if "reasoning" in self.model_setting:
            if not isinstance(self.model_setting["reasoning"], dict):
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Reasoning configuration should be a dictionary. "
                        "Reasoning features may not work correctly."
                    )
            elif self.model_setting["reasoning"].get("enabled") is None:
                if self.logger and self.logger.isEnabledFor(logging.WARNING):
                    self.logger.warning(
                        "Reasoning is not explicitly enabled in configuration. "
                        "Reasoning events will be skipped during streaming."
                    )

        # Pre-build options dict for model invocation (performance optimization)
        option_keys = [
            "temperature",
            "top_p",
            "num_predict",
            "top_k",
            "repeat_penalty",
            "num_ctx",
        ]
        self.model_options = {
            k: self.model_setting[k] for k in option_keys if k in self.model_setting
        }

        # Client uses connection pooling for better performance with multiple requests
        # HTTP/2 is enabled natively for improved performance with multiplexing
        self.client = ollama.Client(
            host=self.model_setting.get("base_url"),
            headers=self.model_setting.get("headers", {}),
            http2=True,
        )

    def _cleanup_input_messages(
        self, input_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filters out broken tool interaction sequences from message history.
        Valid sequences: assistant (with tool_calls) → tool results → assistant (final response).
        Removes tool results without proper initiation or sequences without completion.

        Optimized to O(n) complexity with single-pass algorithm.

        Args:
            input_messages: Raw conversation messages

        Returns:
            Filtered messages containing only valid sequences
        """
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"[_cleanup_input_messages] Cleaning {len(input_messages)} messages"
            )

        if not input_messages:
            return []

        result = []
        tool_call_role = self.agent["tool_call_role"]
        i = 0

        while i < len(input_messages):
            current_msg = input_messages[i]
            current_role = current_msg.get("role")

            # Skip orphaned tool results (not preceded by assistant with tool_calls)
            if current_role == tool_call_role:
                if not (
                    result
                    and result[-1].get("role") == "assistant"
                    and "tool_calls" in result[-1]
                ):
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"[_cleanup_input_messages] Skipping orphaned tool at [{i}]"
                        )
                    i += 1
                    continue
                result.append(current_msg)
                i += 1
                continue

            # Handle assistant messages with tool_calls
            if current_role == "assistant" and "tool_calls" in current_msg:
                # Find the end of tool results sequence
                j = i + 1
                while (
                    j < len(input_messages)
                    and input_messages[j].get("role") == tool_call_role
                ):
                    j += 1

                # Check if sequence is complete (followed by assistant message)
                if (
                    j < len(input_messages)
                    and input_messages[j].get("role") == "assistant"
                ):
                    # Valid sequence: include the tool caller
                    result.append(current_msg)
                    i += 1
                else:
                    # Incomplete sequence: skip entire cycle
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info(
                            f"[_cleanup_input_messages] Skipping incomplete cycle [{i}:{j - 1}]"
                        )
                    i = j
                continue

            # Regular messages (user, assistant without tool_calls)
            result.append(current_msg)
            i += 1

        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(
                f"[_cleanup_input_messages] Retained {len(result)} messages"
            )

        return result

    def _get_elapsed_time(self) -> float:
        """
        Get elapsed time in milliseconds from the first ask_model call.

        Returns:
            Elapsed time in milliseconds, or 0 if global start time not set
        """
        if not hasattr(self, "_global_start_time") or self._global_start_time is None:
            return 0.0
        return (pendulum.now("UTC") - self._global_start_time).total_seconds() * 1000

    def reset_timeline(self) -> None:
        """
        Reset the global timeline for a new run.
        Should be called at the start of each new user interaction/run.
        """
        self._global_start_time = None
        if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
            self.logger.info("[TIMELINE] Timeline reset for new run")

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
            invoke_start = pendulum.now("UTC")
            # Build messages list more efficiently
            messages = [self.system_message] + kwargs.get("input_messages", [])

            # Prepare chat parameters
            chat_params = {
                "model": self.model_setting.get("model"),
                "messages": messages,
            }

            # Use pre-built options from __init__ (performance optimization)
            if self.model_options:
                chat_params["options"] = self.model_options

            # Add tools if available
            if "tools" in self.model_setting:
                chat_params["tools"] = self.model_setting["tools"]

            # Add format option using cached value (performance optimization)
            if self.output_format_type == "json_object":
                chat_params["format"] = "json"
            elif self.output_format_type == "json_schema":
                # Ollama supports json schema format
                format_config = self.model_setting.get("text", {}).get("format", {})
                if "schema" in format_config:
                    chat_params["format"] = format_config["schema"]

            # Add reasoning/thinking parameter if enabled
            reasoning_config = self.model_setting.get("reasoning", {})
            if isinstance(reasoning_config, dict) and reasoning_config.get("enabled"):
                chat_params["think"] = True
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("[invoke_model] Reasoning enabled, think=True")

            # Return streaming or non-streaming response
            if kwargs["stream"]:
                chat_params["stream"] = True

            result = self.client.chat(**chat_params)

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                invoke_end = pendulum.now("UTC")
                invoke_time = (invoke_end - invoke_start).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: API call returned (took {invoke_time:.2f}ms)"
                )

            return result

        except Exception as e:
            if self.logger.isEnabledFor(logging.ERROR):
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
        # Track preparation time
        ask_model_start = pendulum.now("UTC")

        # Track recursion depth to identify top-level vs recursive calls
        if not hasattr(self, "_ask_model_depth"):
            self._ask_model_depth = 0

        self._ask_model_depth += 1
        is_top_level = self._ask_model_depth == 1

        # Initialize global start time only on top-level ask_model call
        # Recursive calls will use the same start time for the entire run timeline
        if is_top_level:
            self._global_start_time = ask_model_start

            # Reset reasoning_summary for new conversation turn
            # Recursive calls (function call loops) will continue accumulating
            if "reasoning_summary" in self.final_output:
                del self.final_output["reasoning_summary"]

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                self.logger.info("[TIMELINE] T+0ms: Run started - First ask_model call")
        else:
            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Recursive ask_model call started"
                )

        try:
            stream = True if queue is not None else False

            # Add model-specific settings if provided
            if model_setting:
                self.model_setting.update(model_setting)

            # Clean up input messages to remove broken tool sequences (performance optimization)
            cleanup_start = pendulum.now("UTC")
            input_messages = self._cleanup_input_messages(input_messages)
            cleanup_end = pendulum.now("UTC")
            cleanup_time = (cleanup_end - cleanup_start).total_seconds() * 1000

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Track total preparation time before API call
                preparation_end = pendulum.now("UTC")
                preparation_time = (
                    preparation_end - ask_model_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Preparation complete (took {preparation_time:.2f}ms, cleanup: {cleanup_time:.2f}ms)"
                )

            timestamp = pendulum.now("UTC").int_timestamp
            # Optimized UUID generation - use .hex instead of str() conversion
            run_id = f"run-ollama-{self.model_setting['model']}-{timestamp}-{uuid.uuid4().hex[:8]}"

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
        finally:
            # Decrement depth when exiting ask_model
            self._ask_model_depth -= 1

            # Reset timeline when returning to depth 0 (top-level call complete)
            if self._ask_model_depth == 0:
                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(
                        f"[TIMELINE] T+{elapsed:.2f}ms: Run complete - Resetting timeline"
                    )
                self._global_start_time = None

    def handle_function_call(
        self, tool_call: Dict[str, Any], input_messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
        # Track function call timing
        function_call_start = pendulum.now("UTC")

        # Ollama format: {"function": {"name": "...", "arguments": {...}}}
        if "function" not in tool_call:
            raise ValueError("Invalid tool_call object: missing 'function' key")

        function_info = tool_call["function"]
        function_call_data = {
            "id": uuid.uuid4().hex,  # Optimized UUID generation
            "arguments": function_info.get("arguments", {}),
            "type": "function",
            "name": function_info["name"],
        }

        try:
            function_name = function_call_data["name"]

            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Starting function call recording for {function_name}"
                )

            self._record_function_call_start(function_call_data)

            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Processing arguments for function {function_name}"
                )

            arguments = self._process_function_arguments(function_call_data)

            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call] Executing function {function_name} with arguments {arguments}"
                )

            function_output = self._execute_function(function_call_data, arguments)

            # Update conversation history
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[handle_function_call][{function_name}] Updating conversation history"
                )

            self._update_conversation_history(
                function_call_data, function_output, input_messages
            )

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

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                # Log function call execution time
                function_call_end = pendulum.now("UTC")
                function_call_time = (
                    function_call_end - function_call_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' complete (took {function_call_time:.2f}ms)"
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

            return arguments

        except Exception as e:
            log = traceback.format_exc()
            # Batch async call with error details (performance optimization)
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": function_call_data.get("arguments", "{}"),
                    "status": "failed",
                    "notes": log,
                },
            )
            if self.logger.isEnabledFor(logging.ERROR):
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
            # Cache JSON serialization to avoid duplicate work (performance optimization)
            arguments_json = Utility.json_dumps(arguments)

            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": arguments_json,
                    "status": "in_progress",
                },
            )

            # Track actual function execution time
            function_exec_start = pendulum.now("UTC")
            function_output = agent_function(**arguments)

            if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                function_exec_end = pendulum.now("UTC")
                function_exec_time = (
                    function_exec_end - function_exec_start
                ).total_seconds() * 1000
                elapsed = self._get_elapsed_time()
                self.logger.info(
                    f"[TIMELINE] T+{elapsed:.2f}ms: Function '{function_call_data['name']}' executed (took {function_exec_time:.2f}ms)"
                )

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
            # Cache JSON serialization to avoid duplicate work (performance optimization)
            arguments_json = Utility.json_dumps(arguments)
            self.invoke_async_funct(
                "async_insert_update_tool_call",
                **{
                    "tool_call_id": function_call_data["id"],
                    "arguments": arguments_json,
                    "status": "failed",
                    "notes": log,
                },
            )
            return f"Function execution failed: {e}"

    def _update_conversation_history(
        self,
        function_call_data: Dict[str, Any],
        function_output: Any,
        input_messages: List[Dict[str, Any]],
    ) -> None:
        """
        Updates the conversation history with function call results.
        Formats and appends function output as a user message.

        Args:
            function_call_data: Metadata about the executed function
            function_output: Result from function execution
            input_messages: Current conversation history to update
        """

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
                "role": self.agent["tool_call_role"],
                "content": content,
                "tool_name": function_call_data["name"],
            }
        )

    def _check_retry_limit(self, retry_count: int) -> None:
        """
        Check if retry limit has been exceeded and raise exception if so.

        Args:
            retry_count: Current retry count

        Raises:
            Exception: If retry_count exceeds MAX_RETRIES
        """
        MAX_RETRIES = 5
        if retry_count > MAX_RETRIES:
            error_msg = (
                f"Maximum retry limit ({MAX_RETRIES}) exceeded for empty responses"
            )
            self.logger.error(error_msg)
            raise Exception(error_msg)

    def _has_valid_content(self, text: str) -> bool:
        """
        Check if response text contains valid content.

        Args:
            text: Response text to check

        Returns:
            True if text is not None/empty/whitespace-only, False otherwise
        """
        return bool(text and text.strip())

    def handle_response(
        self,
        response: Any,
        input_messages: List[Dict[str, Any]],
        retry_count: int = 0,
    ) -> None:
        """
        Processes non-streaming model output.

        Handles three scenarios:
        1. Function call → Execute and recurse
        2. Empty response → Retry up to 5 times
        3. Valid response → Set final_output

        Args:
            response: Model response object (Ollama format)
            input_messages: Current conversation history
            retry_count: Current retry count (max 5 retries)
        """
        self._check_retry_limit(retry_count)

        self.logger.info("Processing output: %s", response)

        # Ollama response format: {"message": {"role": "assistant", "content": "..."}, "model": "...", ...}
        message = response.get("message", {})

        # Extract and store reasoning/thinking if present
        if "thinking" in message and message["thinking"]:
            thinking_text = message["thinking"]
            try:
                if isinstance(thinking_text, str):
                    self.final_output["reasoning_summary"] = thinking_text
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f"[handle_response] Captured reasoning: {thinking_text[:100]}..."
                        )
            except Exception as e:
                if self.logger.isEnabledFor(logging.ERROR):
                    self.logger.error(f"Failed to process reasoning: {e}")
                self.final_output["reasoning_summary"] = "Error processing reasoning"

        # Scenario 1: Handle function calls
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

            # Recurse with fresh response (reset retry count)
            response = self.invoke_model(
                **{"input_messages": input_messages, "stream": False}
            )
            self.handle_response(response, input_messages, retry_count=0)
            return

        # Scenario 2: Empty response - retry
        content = message.get("content", "")
        if not self._has_valid_content(content):
            self.logger.warning(
                f"Received empty response from model, retrying (attempt {retry_count + 1}/5)..."
            )
            next_response = self.invoke_model(
                **{"input_messages": input_messages, "stream": False}
            )
            self.handle_response(
                next_response, input_messages, retry_count=retry_count + 1
            )
            return

        # Scenario 3: Valid response - set final output
        timestamp = pendulum.now("UTC").int_timestamp
        # Optimized UUID generation
        message_id = f"msg-ollama-{self.model_setting.get('model')}-{timestamp}-{uuid.uuid4().hex[:8]}"
        self.final_output.update(
            {
                "message_id": message_id,
                "role": message.get("role", "assistant"),
                "content": content,
            }
        )

    def handle_stream(
        self,
        response_stream: Any,
        input_messages: List[Dict[str, Any]],
        stream_event: threading.Event = None,
        retry_count: int = 0,
    ) -> None:
        """
        Processes streaming model responses chunk by chunk.

        Handles three scenarios:
        1. Function call → Execute and recurse
        2. Empty stream → Retry up to 5 times
        3. Valid stream → Accumulate and set final_output

        Args:
            response_stream: Iterator of response chunks (Ollama format)
            input_messages: Current conversation history
            stream_event: Event to signal completion
            retry_count: Current retry count (max 5 retries)

        Handles:
            - Accumulating response text
            - Processing JSON vs text formats
            - Handling tool calls in streaming chunks
            - Sending chunks to websocket
            - Signaling completion
        """
        self._check_retry_limit(retry_count)

        message_id = None
        # Use list for efficient string concatenation (performance optimization)
        accumulated_text_parts = []
        accumulated_partial_json = ""
        accumulated_partial_text = ""
        accumulated_tool_calls = []
        received_any_content = False
        # Use cached output format type (performance optimization)
        output_format = self.output_format_type
        index = 0

        # Reasoning/thinking tracking variables
        reasoning_no = 0
        reasoning_index = 0
        accumulated_reasoning_parts = []
        accumulated_partial_reasoning_text = ""
        reasoning_started = False

        for chunk in response_stream:
            # Get the message from the chunk
            message = chunk.get("message", {})

            # Check for reasoning/thinking in the chunk
            if "thinking" in message and message["thinking"]:
                thinking_chunk = message["thinking"]
                received_any_content = True

                # Start reasoning block if not started
                if not reasoning_started:
                    reasoning_started = True
                    reasoning_index = 0
                    self.send_data_to_stream(
                        index=reasoning_index,
                        data_format=output_format,
                        chunk_delta=f"<ReasoningStart Id={reasoning_no}/>",
                        suffix=f"rs#{reasoning_no}",
                    )
                    reasoning_index += 1

                    if self.enable_timeline_log and self.logger.isEnabledFor(
                        logging.INFO
                    ):
                        elapsed = self._get_elapsed_time()
                        self.logger.info(
                            f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning started"
                        )

                # Accumulate reasoning text
                print(thinking_chunk, end="", flush=True)
                accumulated_reasoning_parts.append(thinking_chunk)
                accumulated_partial_reasoning_text += thinking_chunk

                # Process and send reasoning text
                reasoning_index, accumulated_partial_reasoning_text = (
                    self.process_text_content(
                        reasoning_index,
                        accumulated_partial_reasoning_text,
                        output_format,
                        suffix=f"rs#{reasoning_no}",
                    )
                )

            # Check if reasoning block has ended (content starts arriving)
            if reasoning_started and "content" in message and message.get("content"):
                # End reasoning block
                if len(accumulated_partial_reasoning_text) > 0:
                    self.send_data_to_stream(
                        index=reasoning_index,
                        data_format=output_format,
                        chunk_delta=accumulated_partial_reasoning_text,
                        suffix=f"rs#{reasoning_no}",
                    )
                    accumulated_partial_reasoning_text = ""
                    reasoning_index += 1

                self.send_data_to_stream(
                    index=reasoning_index,
                    data_format=output_format,
                    chunk_delta=f"<ReasoningEnd Id={reasoning_no}/>",
                    is_message_end=True,
                    suffix=f"rs#{reasoning_no}",
                )
                reasoning_no += 1
                reasoning_started = False

                if self.enable_timeline_log and self.logger.isEnabledFor(logging.INFO):
                    elapsed = self._get_elapsed_time()
                    self.logger.info(f"[TIMELINE] T+{elapsed:.2f}ms: Reasoning ended")

            # Check for tool calls in streaming chunks (Ollama v0.8.0+)
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    # Accumulate tool calls to handle them at the end
                    accumulated_tool_calls.append(tool_call)
                    received_any_content = True

            # Get the content from the chunk
            chunk_content = message.get("content")

            # Skip empty chunks
            if not chunk_content:
                continue

            received_any_content = True

            # Ollama streaming format: {"message": {"role": "assistant", "content": "..."}, "done": false, ...}
            # Generate message_id on first chunk
            if not message_id:
                # Sync index with reasoning_index when starting content after reasoning
                if index == 0 and reasoning_index > 0:
                    index = reasoning_index

                self.send_data_to_stream(
                    index=index,
                    data_format=output_format,
                )
                index += 1

                timestamp = pendulum.now("UTC").int_timestamp
                # Optimized UUID generation
                message_id = f"msg-ollama-{self.model_setting.get('model')}-{timestamp}-{uuid.uuid4().hex[:8]}"

            # Print out for stream and accumulate in list (performance optimization)
            print(chunk_content, end="", flush=True)
            accumulated_text_parts.append(chunk_content)

            if output_format in ["json_object", "json_schema"]:
                accumulated_partial_json += chunk_content
                # Temporarily build accumulated_text for processing
                temp_accumulated_text = "".join(accumulated_text_parts)
                index, temp_accumulated_text, accumulated_partial_json = (
                    self.process_and_send_json(
                        index,
                        temp_accumulated_text,
                        accumulated_partial_json,
                        output_format,
                    )
                )
            else:
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

        # Build final accumulated text from parts (performance optimization)
        final_accumulated_text = "".join(accumulated_text_parts)

        # Store accumulated reasoning summary if present
        if accumulated_reasoning_parts:
            final_reasoning_text = "".join(accumulated_reasoning_parts)
            if self.final_output.get("reasoning_summary"):
                # Accumulate reasoning from multiple rounds (e.g., function calls)
                self.final_output["reasoning_summary"] = (
                    self.final_output["reasoning_summary"] + "\n" + final_reasoning_text
                )
            else:
                self.final_output["reasoning_summary"] = final_reasoning_text

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[handle_stream] Stored reasoning summary: {final_reasoning_text[:100]}..."
                )

        # Scenario 1: Handle accumulated tool calls after streaming completes
        if accumulated_tool_calls:
            # Append the assistant's message with tool_calls
            input_messages.append(
                {
                    "role": "assistant",
                    "content": final_accumulated_text,
                    "tool_calls": accumulated_tool_calls,
                }
            )

            # Then, append tool results
            for tool_call in accumulated_tool_calls:
                input_messages = self.handle_function_call(tool_call, input_messages)

            # Recurse with fresh response (reset retry count)
            response = self.invoke_model(
                **{"input_messages": input_messages, "stream": True}
            )
            self.handle_stream(
                response, input_messages, stream_event=stream_event, retry_count=0
            )
            return

        # Scenario 2: Empty stream - retry
        if not received_any_content:
            self.logger.warning(
                f"Received empty response from model, retrying (attempt {retry_count + 1}/5)..."
            )
            next_response = self.invoke_model(
                **{"input_messages": input_messages, "stream": True}
            )
            self.handle_stream(
                next_response,
                input_messages,
                stream_event=stream_event,
                retry_count=retry_count + 1,
            )
            return

        # Scenario 3: Valid stream - finalize
        # Send final message end signal
        self.send_data_to_stream(
            index=index,
            data_format=output_format,
            is_message_end=True,
        )

        # Set final output
        self.final_output.update(
            {
                "message_id": (
                    message_id
                    if message_id
                    # Optimized UUID generation
                    else f"msg-{pendulum.now('UTC').int_timestamp}-{uuid.uuid4().hex[:8]}"
                ),
                "role": "assistant",
                "content": final_accumulated_text,
            }
        )

        # Store accumulated_text for backward compatibility
        self.accumulated_text = final_accumulated_text

        # Signal that streaming has finished
        if stream_event:
            stream_event.set()
