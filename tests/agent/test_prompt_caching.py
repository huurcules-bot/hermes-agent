"""Tests for agent/prompt_caching.py — Anthropic cache control injection."""

import copy
import pytest

from agent.prompt_caching import (
    _apply_cache_marker,
    apply_anthropic_cache_control,
    apply_anthropic_tools_cache_control,
)


MARKER = {"type": "ephemeral"}


class TestApplyCacheMarker:
    def test_tool_message_gets_top_level_marker_on_native_anthropic(self):
        """Native Anthropic path: cache_control injected top-level (adapter moves it inside tool_result)."""
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=True)
        assert msg["cache_control"] == MARKER

    def test_tool_message_skips_marker_on_openrouter(self):
        """OpenRouter path: top-level cache_control on role:tool is invalid and causes silent hang."""
        msg = {"role": "tool", "content": "result"}
        _apply_cache_marker(msg, MARKER, native_anthropic=False)
        assert "cache_control" not in msg

    def test_none_content_gets_top_level_marker(self):
        msg = {"role": "assistant", "content": None}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER

    def test_empty_string_content_gets_top_level_marker(self):
        """Empty text blocks cannot have cache_control (Anthropic rejects them)."""
        msg = {"role": "assistant", "content": ""}
        _apply_cache_marker(msg, MARKER)
        assert msg["cache_control"] == MARKER
        # Must NOT wrap into [{"type": "text", "text": "", "cache_control": ...}]
        assert msg["content"] == ""

    def test_string_content_wrapped_in_list(self):
        msg = {"role": "user", "content": "Hello"}
        _apply_cache_marker(msg, MARKER)
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello"
        assert msg["content"][0]["cache_control"] == MARKER

    def test_list_content_last_item_gets_marker(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ],
        }
        _apply_cache_marker(msg, MARKER)
        assert "cache_control" not in msg["content"][0]
        assert msg["content"][1]["cache_control"] == MARKER

    def test_empty_list_content_no_crash(self):
        msg = {"role": "user", "content": []}
        # Should not crash on empty list
        _apply_cache_marker(msg, MARKER)


class TestApplyAnthropicCacheControl:
    def test_empty_messages(self):
        result = apply_anthropic_cache_control([])
        assert result == []

    def test_returns_deep_copy(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = apply_anthropic_cache_control(msgs)
        assert result is not msgs
        assert result[0] is not msgs[0]
        # Original should be unmodified
        assert "cache_control" not in msgs[0].get("content", "")

    def test_marks_only_last_message_when_no_system_reminders(self):
        msgs = [
            {"role": "user", "content": "msg1"},
            {"role": "assistant", "content": "msg2"},
            {"role": "user", "content": "msg3"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # Earlier messages: no marker
        for i in range(2):
            content = result[i]["content"]
            if isinstance(content, list):
                assert "cache_control" not in content[-1]
        # Last message: marker
        last_content = result[-1]["content"]
        assert isinstance(last_content, list)
        assert last_content[-1]["cache_control"] == {"type": "ephemeral"}

    def test_marks_every_system_reminder_message(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>\nstable\n</system-reminder>"}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>\nephemeral\n</system-reminder>"}
                ],
            },
            {"role": "user", "content": "actual prompt"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # Both <system-reminder> messages get markers.
        assert result[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        assert result[1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        # The last message also gets a marker.
        last_content = result[-1]["content"]
        assert isinstance(last_content, list)
        assert last_content[-1]["cache_control"] == {"type": "ephemeral"}

    def test_does_not_mark_non_reminder_messages_in_the_middle(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>\nstable\n</system-reminder>"}
                ],
            },
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second turn"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # The reminder message is marked.
        assert result[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
        # Middle non-reminder messages are NOT marked.
        for i in (1, 2):
            content = result[i]["content"]
            if isinstance(content, list):
                assert "cache_control" not in content[-1]
        # The last message is marked.
        assert isinstance(result[-1]["content"], list)
        assert result[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}

    def test_recognises_string_content_starting_with_system_reminder(self):
        msgs = [
            {"role": "user", "content": "<system-reminder>\nstable\n</system-reminder>"},
            {"role": "user", "content": "actual prompt"},
        ]
        result = apply_anthropic_cache_control(msgs)
        # String content gets normalized into a list with the marker.
        first_content = result[0]["content"]
        assert isinstance(first_content, list)
        assert first_content[-1]["cache_control"] == {"type": "ephemeral"}

    def test_last_message_marked_once_when_it_is_a_system_reminder(self):
        # Only one message and it IS a system-reminder — it should still
        # only carry a single marker.
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>\nonly\n</system-reminder>"}
                ],
            },
        ]
        result = apply_anthropic_cache_control(msgs)
        block = result[0]["content"][-1]
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_1h_ttl(self):
        msgs = [
            {"role": "user", "content": "Hello"},
        ]
        result = apply_anthropic_cache_control(msgs, cache_ttl="1h")
        last = result[-1]["content"]
        assert isinstance(last, list)
        assert last[-1]["cache_control"]["ttl"] == "1h"

    def test_at_most_one_marker_per_message_block(self):
        # Two reminders + last message = 3 total markers max here.
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>\nstable\n</system-reminder>"}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<system-reminder>\nephemeral\n</system-reminder>"}
                ],
            },
            {"role": "user", "content": "tail"},
        ]
        result = apply_anthropic_cache_control(msgs)
        count = 0
        for msg in result:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "cache_control" in item:
                        count += 1
            elif "cache_control" in msg:
                count += 1
        assert count == 3


class TestApplyAnthropicToolsCacheControl:
    def test_none_returns_none(self):
        assert apply_anthropic_tools_cache_control(None) is None

    def test_empty_list_returns_input_unchanged(self):
        tools = []
        assert apply_anthropic_tools_cache_control(tools) is tools

    def test_single_tool_gets_marker(self):
        tools = [{"name": "Bash", "description": "shell", "input_schema": {}}]
        result = apply_anthropic_tools_cache_control(tools)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_multi_tool_only_last_gets_marker(self):
        tools = [
            {"name": "Bash", "description": "shell", "input_schema": {}},
            {"name": "Read", "description": "read file", "input_schema": {}},
            {"name": "Write", "description": "write file", "input_schema": {}},
        ]
        result = apply_anthropic_tools_cache_control(tools)
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]
        assert result[2]["cache_control"] == {"type": "ephemeral"}

    def test_1h_ttl_sets_ttl_field(self):
        tools = [{"name": "Bash", "description": "shell", "input_schema": {}}]
        result = apply_anthropic_tools_cache_control(tools, cache_ttl="1h")
        assert result[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_returns_deep_copy(self):
        tools = [{"name": "Bash", "description": "shell", "input_schema": {}}]
        result = apply_anthropic_tools_cache_control(tools)
        # Mutations to the result must not bleed into the input.
        assert result is not tools
        assert result[0] is not tools[0]
        assert "cache_control" not in tools[0]
