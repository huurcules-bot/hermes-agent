"""Anthropic prompt caching.

Reduces input token costs by placing ``cache_control: ephemeral``
breakpoints at stable cut-points in the request. Anthropic allows up to 4
breakpoints; this module places message-level ones. Tools-list breakpoint
lives in ``apply_anthropic_tools_cache_control``.

``apply_anthropic_cache_control`` marks two kinds of messages:
  - every text content block that starts with ``<system-reminder>``
    (each system-reminder block gets its own breakpoint)
  - the very last message in ``messages`` (rolling tail)

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List, Optional


_SYSTEM_REMINDER_PREFIX = "<system-reminder>"


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> List[Dict[str, Any]]:
    """Mark cache breakpoints on ``<system-reminder>`` content blocks and the last message.

    For each message, iterates every content block and adds ``cache_control``
    to each text block whose text starts with ``<system-reminder>``. Plain
    string content is converted to a block first. Also marks the final message
    as a rolling-tail breakpoint.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            if content.startswith(_SYSTEM_REMINDER_PREFIX):
                _apply_cache_marker(msg, marker, native_anthropic=native_anthropic)
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and block["text"].startswith(_SYSTEM_REMINDER_PREFIX)
                ):
                    block["cache_control"] = marker

    _apply_cache_marker(messages[-1], marker, native_anthropic=native_anthropic)

    return messages


def apply_anthropic_tools_cache_control(
    tools: Optional[List[Dict[str, Any]]],
    cache_ttl: str = "5m",
) -> Optional[List[Dict[str, Any]]]:
    """Mark the last tool with ``cache_control: ephemeral``.

    Anthropic caches up to and including the marked block, so a single mark
    on the final tool keeps the entire tools list in the cached prefix.
    Tool schemas are stable across all turns in a session, so this trades
    one ephemeral cache slot for amortising the (often very large) tool
    schema cost across the whole session.

    Returns a deep copy with the marker on the last tool. Returns the input
    unchanged when ``tools`` is None or empty (no tool to mark).
    """
    if not tools:
        return tools
    out = copy.deepcopy(tools)
    marker: Dict[str, Any] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"
    last = out[-1]
    if isinstance(last, dict):
        last["cache_control"] = marker
    return out
