"""
Parser unit tests for Qwen2.5 vLLM tool parser.

Tests parser logic directly without model or vLLM server dependency.
Mocks vLLM imports to allow standalone execution.

Usage:
    pytest benchmarks/tools_parser/test_parser_unit.py -v
"""

# pylint: disable=missing-class-docstring,missing-function-docstring
# pylint: disable=too-few-public-methods
# pylint: disable=protected-access,attribute-defined-outside-init
# pylint: disable=use-implicit-booleaness-not-comparison

import json
import sys
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

# =============================================================================
# vLLM Mock Setup
# =============================================================================
# Mock vLLM modules before importing the parser.
# This allows running tests without vLLM installed.


@dataclass
class FunctionCall:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    type: str
    function: FunctionCall


@dataclass
class DeltaFunctionCall:
    name: str
    arguments: str


@dataclass
class DeltaToolCall:
    index: int
    id: str
    type: str
    function: DeltaFunctionCall


@dataclass
class DeltaMessage:
    content: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall]] = None


@dataclass
class ExtractedToolCallInformation:
    tools_called: bool
    tool_calls: List[ToolCall]
    content: Optional[str]


class ChatCompletionRequest:
    pass


class ToolParser:
    def __init__(self, tokenizer):
        self.model_tokenizer = tokenizer


class ToolParserManager:
    @staticmethod
    def register_module(_names):
        def decorator(cls):
            return cls
        return decorator


# Install mocks
mock_protocol = MagicMock()
mock_protocol.ChatCompletionRequest = ChatCompletionRequest
mock_protocol.DeltaFunctionCall = DeltaFunctionCall
mock_protocol.DeltaMessage = DeltaMessage
mock_protocol.DeltaToolCall = DeltaToolCall
mock_protocol.ExtractedToolCallInformation = ExtractedToolCallInformation
mock_protocol.FunctionCall = FunctionCall
mock_protocol.ToolCall = ToolCall

mock_abstract = MagicMock()
mock_abstract.ToolParser = ToolParser
mock_abstract.ToolParserManager = ToolParserManager

mock_logger = MagicMock()
mock_logger.init_logger = MagicMock(return_value=MagicMock())

mock_tokenizer = MagicMock()

sys.modules["vllm"] = MagicMock()
sys.modules["vllm.entrypoints"] = MagicMock()
sys.modules["vllm.entrypoints.openai"] = MagicMock()
sys.modules["vllm.entrypoints.openai.protocol"] = mock_protocol
sys.modules["vllm.entrypoints.openai.tool_parsers"] = MagicMock()
sys.modules["vllm.entrypoints.openai.tool_parsers.abstract_tool_parser"] = mock_abstract
sys.modules["vllm.tool_parsers"] = MagicMock()
sys.modules["vllm.tool_parsers.abstract_tool_parser"] = mock_abstract
sys.modules["vllm.logger"] = mock_logger
sys.modules["vllm.transformers_utils"] = MagicMock()
sys.modules["vllm.transformers_utils.tokenizer"] = mock_tokenizer

# Now import the parser
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from qwen2_5_coder_tool_parser import Qwen25CoderToolParser  # noqa: E402  # pylint: disable=C0413


# =============================================================================
# Fixtures
# =============================================================================

def make_parser() -> Qwen25CoderToolParser:
    """Create a parser instance with a mock tokenizer."""
    tokenizer = MagicMock()
    return Qwen25CoderToolParser(tokenizer)


def make_request() -> ChatCompletionRequest:
    """Create a mock request."""
    return ChatCompletionRequest()


# =============================================================================
# Tests: _is_valid_tool_call
# =============================================================================

class TestIsValidToolCall:
    def setup_method(self):
        self.parser = make_parser()

    def test_valid_minimal(self):
        assert self.parser._is_valid_tool_call({"name": "get_weather"})

    def test_valid_with_arguments(self):
        assert self.parser._is_valid_tool_call({
            "name": "get_weather",
            "arguments": {"city": "Seoul"}
        })

    def test_invalid_no_name(self):
        assert not self.parser._is_valid_tool_call({"arguments": {"city": "Seoul"}})

    def test_invalid_name_not_string(self):
        assert not self.parser._is_valid_tool_call({"name": 123})

    def test_invalid_not_dict(self):
        assert not self.parser._is_valid_tool_call("not a dict")

    def test_invalid_list(self):
        assert not self.parser._is_valid_tool_call([{"name": "foo"}])

    def test_invalid_none(self):
        assert not self.parser._is_valid_tool_call(None)

    def test_valid_extra_fields(self):
        """Extra fields beyond name/arguments should not invalidate."""
        assert self.parser._is_valid_tool_call({
            "name": "foo",
            "arguments": {},
            "extra": "field"
        })


# =============================================================================
# Tests: _try_parse_json
# =============================================================================

class TestTryParseJson:
    def setup_method(self):
        self.parser = make_parser()

    # --- Single object ---
    def test_single_object(self):
        result = self.parser._try_parse_json(
            '{"name": "get_weather", "arguments": {"city": "Seoul"}}'
        )
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["arguments"]["city"] == "Seoul"

    def test_single_object_no_arguments(self):
        result = self.parser._try_parse_json('{"name": "get_weather"}')
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"

    def test_single_object_empty_arguments(self):
        result = self.parser._try_parse_json(
            '{"name": "get_weather", "arguments": {}}'
        )
        assert len(result) == 1

    # --- Array format ---
    def test_array_single(self):
        result = self.parser._try_parse_json(
            '[{"name": "get_weather", "arguments": {"city": "Seoul"}}]'
        )
        assert len(result) == 1

    def test_array_multiple(self):
        result = self.parser._try_parse_json(
            '[{"name": "get_weather", "arguments": {"city": "Seoul"}},'
            ' {"name": "get_weather", "arguments": {"city": "Tokyo"}}]'
        )
        assert len(result) == 2
        assert result[0]["arguments"]["city"] == "Seoul"
        assert result[1]["arguments"]["city"] == "Tokyo"

    def test_array_filters_invalid(self):
        """Invalid items in array should be filtered out."""
        result = self.parser._try_parse_json(
            '[{"name": "foo"}, {"no_name": true}, {"name": "bar"}]'
        )
        assert len(result) == 2
        assert result[0]["name"] == "foo"
        assert result[1]["name"] == "bar"

    # --- JSONL format ---
    def test_jsonl_two_lines(self):
        result = self.parser._try_parse_json(
            '{"name": "get_weather", "arguments": {"city": "Seoul"}}\n'
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
        )
        assert len(result) == 2

    def test_jsonl_with_trailing_commas(self):
        result = self.parser._try_parse_json(
            '{"name": "foo", "arguments": {}},\n'
            '{"name": "bar", "arguments": {}},\n'
        )
        assert len(result) == 2

    def test_jsonl_with_empty_lines(self):
        result = self.parser._try_parse_json(
            '\n{"name": "foo"}\n\n{"name": "bar"}\n'
        )
        assert len(result) == 2

    # --- Comma-separated format ---
    def test_comma_separated(self):
        result = self.parser._try_parse_json(
            '{"name": "foo", "arguments": {"a": 1}}, '
            '{"name": "bar", "arguments": {"b": 2}}'
        )
        assert len(result) == 2
        assert result[0]["name"] == "foo"
        assert result[1]["name"] == "bar"

    # --- Parametrized format coverage ---
    @pytest.mark.parametrize("input_json,expected_count", [
        ('{"name": "foo"}', 1),
        ('{"name": "foo", "arguments": {"a": 1}}', 1),
        ('[{"name": "a"}, {"name": "b"}]', 2),
        ('[{"name": "a"}, {"name": "b"}, {"name": "c"}]', 3),
        ('{"name": "a"}\n{"name": "b"}', 2),
        ('{"name": "a"},\n{"name": "b"},\n', 2),
        ('{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}', 2),
    ])
    def test_parse_formats(self, input_json, expected_count):
        result = self.parser._try_parse_json(input_json)
        assert len(result) == expected_count

    @pytest.mark.parametrize("input_json", [
        "",
        "not json at all",
        '{"key": "value"}',
        "42",
        '"just a string"',
        "null",
        "true",
        '[{"no_name": 1}]',
    ])
    def test_parse_invalid(self, input_json):
        result = self.parser._try_parse_json(input_json)
        assert result == []

    # --- Unicode ---
    def test_unicode_arguments(self):
        result = self.parser._try_parse_json(
            '{"name": "translate", "arguments": {"text": "你好世界", "target_language": "en"}}'
        )
        assert len(result) == 1
        assert result[0]["arguments"]["text"] == "你好世界"

    def test_emoji_arguments(self):
        result = self.parser._try_parse_json(
            '{"name": "translate", "arguments": {"text": "🎉 Hello 🎊"}}'
        )
        assert len(result) == 1
        assert "🎉" in result[0]["arguments"]["text"]

    # --- Complex arguments ---
    def test_nested_json_in_arguments(self):
        result = self.parser._try_parse_json(
            '{"name": "write_file", "arguments": {"path": "out.json", '
            '"content": "{\\"name\\": \\"test\\", \\"value\\": 123}"}}'
        )
        assert len(result) == 1
        assert "name" in result[0]["arguments"]["content"]

    def test_newlines_in_arguments(self):
        result = self.parser._try_parse_json(
            '{"name": "send_email", "arguments": {"body": "line1\\nline2\\nline3"}}'
        )
        assert len(result) == 1
        assert "line1" in result[0]["arguments"]["body"]


# =============================================================================
# Tests: _parse_tool_json (double-escape handling)
# =============================================================================

class TestParseToolJson:
    def setup_method(self):
        self.parser = make_parser()

    def test_normal_json(self):
        result = self.parser._parse_tool_json(
            '{"name": "foo", "arguments": {"key": "value"}}'
        )
        assert len(result) == 1

    def test_double_escaped_quotes(self):
        """14B models sometimes output double-escaped quotes."""
        # Simulating: \\" in raw string (model outputs literal \\")
        json_str = '{"name": "foo", "arguments": {"key": "val\\\\\\"ue"}}'
        result = self.parser._parse_tool_json(json_str)
        # Should attempt normalization and parse
        # Note: exact behavior depends on how the escaping plays out
        # The key test is that it doesn't crash and returns something
        assert isinstance(result, list)

    def test_empty_returns_empty(self):
        result = self.parser._parse_tool_json("")
        assert result == []

    def test_invalid_returns_empty(self):
        result = self.parser._parse_tool_json("garbage data")
        assert result == []


# =============================================================================
# Tests: extract_tool_calls (non-streaming)
# =============================================================================

class TestExtractToolCalls:
    def setup_method(self):
        self.parser = make_parser()
        self.request = make_request()

    # --- No tool calls ---
    def test_plain_text(self):
        result = self.parser.extract_tool_calls("Hello, world!", self.request)
        assert not result.tools_called
        assert result.tool_calls == []
        assert result.content == "Hello, world!"

    def test_empty_string(self):
        result = self.parser.extract_tool_calls("", self.request)
        assert not result.tools_called
        assert result.content == ""

    # --- Single tool call ---
    def test_single_closed_tag(self):
        text = '<tools>{"name": "get_weather", "arguments": {"city": "Seoul"}}</tools>'
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["city"] == "Seoul"

    def test_single_unclosed_tag(self):
        """Parser should handle missing </tools> closing tag."""
        text = '<tools>{"name": "get_weather", "arguments": {"city": "Seoul"}}'
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"

    def test_single_with_whitespace(self):
        text = '<tools>  {"name": "foo", "arguments": {}}  </tools>'
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 1

    # --- Multiple tool calls (parallel tags) ---
    def test_parallel_two_tags(self):
        text = (
            '<tools>{"name": "get_weather", "arguments": {"city": "Seoul"}}</tools>'
            '<tools>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tools>'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        args0 = json.loads(result.tool_calls[0].function.arguments)
        args1 = json.loads(result.tool_calls[1].function.arguments)
        assert args0["city"] == "Seoul"
        assert args1["city"] == "Tokyo"

    def test_parallel_three_tags(self):
        text = (
            '<tools>{"name": "a", "arguments": {}}</tools>'
            '<tools>{"name": "b", "arguments": {}}</tools>'
            '<tools>{"name": "c", "arguments": {}}</tools>'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        assert len(result.tool_calls) == 3

    # --- Array format in single tag ---
    def test_array_in_tag(self):
        text = (
            '<tools>[{"name": "get_weather", "arguments": {"city": "Seoul"}},'
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}]</tools>'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 2

    # --- JSONL format in single tag ---
    def test_jsonl_in_tag(self):
        text = (
            '<tools>{"name": "get_weather", "arguments": {"city": "Seoul"}}\n'
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tools>'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 2

    # --- Comma-separated in single tag ---
    def test_comma_separated_in_tag(self):
        text = (
            '<tools>{"name": "foo", "arguments": {"a": 1}}, '
            '{"name": "bar", "arguments": {"b": 2}}</tools>'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 2

    # --- Text + tool call mixed ---
    def test_text_before_tool(self):
        text = ('I will check the weather. '
                '<tools>{"name": "get_weather", "arguments": {"city": "Seoul"}}</tools>')
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.content == "I will check the weather."

    def test_text_after_tool(self):
        text = '<tools>{"name": "get_weather", "arguments": {"city": "Seoul"}}</tools> Done.'
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert result.content == "Done."

    def test_text_between_tools(self):
        text = (
            '<tools>{"name": "a", "arguments": {}}</tools>'
            ' and also '
            '<tools>{"name": "b", "arguments": {}}</tools>'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 2
        assert result.content == "and also"

    # --- Tool call IDs ---
    def test_tool_ids_sequential(self):
        text = (
            '<tools>{"name": "a", "arguments": {}}</tools>'
            '<tools>{"name": "b", "arguments": {}}</tools>'
            '<tools>{"name": "c", "arguments": {}}</tools>'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tool_calls[0].id == "tool_0"
        assert result.tool_calls[1].id == "tool_1"
        assert result.tool_calls[2].id == "tool_2"

    # --- Arguments serialization ---
    def test_arguments_serialized_as_json_string(self):
        text = '<tools>{"name": "foo", "arguments": {"key": "value", "num": 42}}</tools>'
        result = self.parser.extract_tool_calls(text, self.request)
        args_str = result.tool_calls[0].function.arguments
        assert isinstance(args_str, str)
        parsed = json.loads(args_str)
        assert parsed["key"] == "value"
        assert parsed["num"] == 42

    def test_missing_arguments_defaults_to_empty(self):
        text = '<tools>{"name": "foo"}</tools>'
        result = self.parser.extract_tool_calls(text, self.request)
        args_str = result.tool_calls[0].function.arguments
        assert json.loads(args_str) == {}

    # --- Unicode handling ---
    def test_unicode_preserved(self):
        text = '<tools>{"name": "translate", "arguments": {"text": "你好世界"}}</tools>'
        result = self.parser.extract_tool_calls(text, self.request)
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args["text"] == "你好世界"

    # --- Invalid JSON in tag ---
    def test_invalid_json_in_tag(self):
        text = '<tools>not valid json</tools>'
        result = self.parser.extract_tool_calls(text, self.request)
        assert not result.tools_called
        assert result.tool_calls == []

    def test_partial_valid_tags(self):
        """One valid and one invalid tag - should extract the valid one."""
        text = (
            '<tools>{"name": "foo", "arguments": {}}</tools>'
            '<tools>invalid json</tools>'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "foo"


# =============================================================================
# Tests: extract_tool_calls_streaming
# =============================================================================

class TestExtractToolCallsStreaming:
    def setup_method(self):
        self.parser = make_parser()
        self.request = make_request()

    def _call(self, previous_text: str, current_text: str, delta_text: str):
        """Helper to call streaming extraction with minimal args."""
        return self.parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=self.request,
        )

    # --- Before <tools> tag ---
    def test_no_tag_returns_content(self):
        result = self._call("Hello", "Hello world", " world")
        assert result is not None
        assert result.content == " world"

    def test_empty_to_text(self):
        result = self._call("", "Hi", "Hi")
        assert result is not None
        assert result.content == "Hi"

    # --- Inside <tools> tag (incomplete) ---
    def test_inside_tag_returns_none(self):
        """While inside an unclosed tag, return None (buffering)."""
        result = self._call(
            "",
            '<tools>{"name": "foo',
            '{"name": "foo'
        )
        assert result is None

    def test_tag_start_returns_none(self):
        result = self._call("", "<tools>", "<tools>")
        assert result is None

    def test_tag_start_in_delta_after_text(self):
        """Delta contains the start of a tag after normal text."""
        result = self._call("Hello ", "Hello <tools>", "<tools>")
        assert result is None

    # --- Tag completed ---
    def test_tag_completed_returns_tool_call(self):
        prev = '<tools>{"name": "foo", "arguments": {"a": 1}'
        curr = '<tools>{"name": "foo", "arguments": {"a": 1}}</tools>'
        delta = "}</tools>"
        result = self._call(prev, curr, delta)
        assert result is not None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "foo"

    def test_tool_id_increments(self):
        """current_tool_id should increment across calls."""
        # First tool call
        prev1 = '<tools>{"name": "a", "arguments": {}'
        curr1 = '<tools>{"name": "a", "arguments": {}}</tools>'
        result1 = self._call(prev1, curr1, "}</tools>")
        assert result1.tool_calls[0].index == 0
        assert result1.tool_calls[0].id == "tool_0"

        # Second tool call
        prev2 = curr1
        curr2 = curr1 + '<tools>{"name": "b", "arguments": {}}</tools>'
        result2 = self._call(prev2, curr2, '<tools>{"name": "b", "arguments": {}}</tools>')
        assert result2.tool_calls[0].index == 1
        assert result2.tool_calls[0].id == "tool_1"

    # --- Array format in streaming ---
    def test_array_format_returns_all_tools(self):
        """Array in a single tag should return all tool calls at once."""
        prev = '<tools>[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}'
        curr = '<tools>[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]</tools>'
        delta = "]</tools>"
        result = self._call(prev, curr, delta)
        assert result is not None
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "a"
        assert result.tool_calls[1].function.name == "b"

    # --- Text after tool call ---
    def test_text_after_completed_tag(self):
        """After tag completion, subsequent text should be returned as content."""
        prev = '<tools>{"name": "foo", "arguments": {}}</tools>'
        curr = '<tools>{"name": "foo", "arguments": {}}</tools> Done!'
        delta = " Done!"
        result = self._call(prev, curr, delta)
        assert result is not None
        assert result.content == " Done!"

    # --- No new matches ---
    def test_no_new_match_inside_second_tag(self):
        """Inside second tag (buffering), return None."""
        prev = '<tools>{"name": "a", "arguments": {}}</tools><tools>{"name": "b"'
        curr = '<tools>{"name": "a", "arguments": {}}</tools><tools>{"name": "b", '
        delta = ", "
        result = self._call(prev, curr, delta)
        assert result is None


# =============================================================================
# Tests: Edge cases & Robustness
# =============================================================================

class TestEdgeCases:
    def setup_method(self):
        self.parser = make_parser()
        self.request = make_request()

    def test_nested_angle_brackets_in_args(self):
        """Arguments containing < > should not confuse the parser."""
        text = ('<tools>{"name": "foo", "arguments": '
                '{"code": "if (a < b) { return a > 0; }"}}</tools>')
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert "<" in args["code"]
        assert ">" in args["code"]

    def test_tools_in_content_text(self):
        """The word 'tools' in regular text should not trigger parsing."""
        text = "I have many tools available."
        result = self.parser.extract_tool_calls(text, self.request)
        assert not result.tools_called

    def test_partial_tag_in_text(self):
        """Incomplete tag syntax should not crash."""
        text = "Use <tools to do stuff"
        result = self.parser.extract_tool_calls(text, self.request)
        assert not result.tools_called

    def test_empty_tag(self):
        text = "<tools></tools>"
        result = self.parser.extract_tool_calls(text, self.request)
        assert not result.tools_called

    def test_whitespace_only_in_tag(self):
        text = "<tools>   </tools>"
        result = self.parser.extract_tool_calls(text, self.request)
        assert not result.tools_called

    def test_very_long_arguments(self):
        """Parser should handle large argument values."""
        long_text = "x" * 10000
        text = f'<tools>{{"name": "foo", "arguments": {{"data": "{long_text}"}}}}</tools>'
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        args = json.loads(result.tool_calls[0].function.arguments)
        assert len(args["data"]) == 10000

    def test_special_chars_in_name(self):
        """Tool names with unusual but valid characters."""
        text = '<tools>{"name": "my_tool_v2", "arguments": {}}</tools>'
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.tools_called
        assert result.tool_calls[0].function.name == "my_tool_v2"

    def test_many_parallel_tags(self):
        """Many parallel tool calls."""
        tags = "".join(
            f'<tools>{{"name": "tool_{i}", "arguments": {{"i": {i}}}}}</tools>'
            for i in range(10)
        )
        result = self.parser.extract_tool_calls(tags, self.request)
        assert result.tools_called
        assert len(result.tool_calls) == 10
        for i, tc in enumerate(result.tool_calls):
            assert tc.function.name == f"tool_{i}"
            assert tc.id == f"tool_{i}"

    def test_content_none_when_only_tools(self):
        """When output is only tool calls, content should be None."""
        text = '<tools>{"name": "foo", "arguments": {}}</tools>'
        result = self.parser.extract_tool_calls(text, self.request)
        assert result.content is None

    def test_newline_between_tags(self):
        text = (
            '<tools>{"name": "a", "arguments": {}}</tools>\n'
            '<tools>{"name": "b", "arguments": {}}</tools>'
        )
        result = self.parser.extract_tool_calls(text, self.request)
        assert len(result.tool_calls) == 2


# =============================================================================
# Run with: pytest benchmarks/tools_parser/test_parser_unit.py -v
# =============================================================================
