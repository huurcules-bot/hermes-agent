"""Anthropic prompt caching.

Reduces input token costs by placing ``cache_control: ephemeral``
breakpoints at stable cut-points in the request. Anthropic allows up to 4
breakpoints; this module places message-level ones. Tools-list breakpoint
lives in ``apply_anthropic_tools_cache_control``.

``apply_anthropic_cache_control`` marks two kinds of content blocks:
  - up to the last 2 text content blocks that start with ``<system-reminder>``
    (gated to 2 to stay within the 4-breakpoint limit)
  - (rolling tail is now handled automatically via top-level cache_control
    set by ``apply_request_level_cache_control`` — no explicit last-message
    marker is placed here)

Breakpoint budget: 1 (tools) + up to 2 (system-reminders) + 1 (automatic
rolling tail from top-level cache_control) = 4 max.

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
    """Mark cache breakpoints on up to the last 2 ``<system-reminder>`` content blocks.

    For each message, iterates every content block and collects references to
    text blocks whose text starts with ``<system-reminder>``. Then marks only
    the last 2 such blocks (to stay within the 4-breakpoint budget: 1 tools +
    2 system-reminders + 1 automatic rolling tail).

    Plain string content that starts with ``<system-reminder>`` is converted
    to a block first during the marking step.

    Note: The rolling-tail (last message) breakpoint is NO LONGER placed here.
    Use ``apply_request_level_cache_control`` to set a top-level cache_control
    on the API kwargs instead — the API handles rolling-tail automatically.

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"

    # First pass: collect all system-reminder block references.
    # Each entry is a dict block (already in list-content form).
    # For string content we handle during the marking pass below.
    reminder_blocks: List[Dict[str, Any]] = []
    string_reminder_msgs: List[Dict[str, Any]] = []

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            if content.startswith(_SYSTEM_REMINDER_PREFIX):
                string_reminder_msgs.append(msg)
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and block["text"].startswith(_SYSTEM_REMINDER_PREFIX)
                ):
                    reminder_blocks.append(block)

    # Combine: string-content reminders count as one block each.
    # We need to gate the total to 2. Process in order: string reminders
    # come from earlier passes; for simplicity, count all reminders together.
    total_reminders = len(string_reminder_msgs) + len(reminder_blocks)
    # Keep only the last 2 from the combined list (in document order,
    # string_reminder_msgs appear interleaved; we approximate by taking
    # last 2 from each category proportionally — but since bypass module
    # creates list-content blocks, the common case is all in reminder_blocks).
    # Simple approach: mark last 2 overall. String-content reminders are
    # rare so we process list-content blocks first (they appear in order).
    skip_count = max(0, total_reminders - 2)

    skipped = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            if content.startswith(_SYSTEM_REMINDER_PREFIX):
                if skipped < skip_count:
                    skipped += 1
                else:
                    _apply_cache_marker(msg, marker, native_anthropic=native_anthropic)
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and block["text"].startswith(_SYSTEM_REMINDER_PREFIX)
                ):
                    if skipped < skip_count:
                        skipped += 1
                    else:
                        block["cache_control"] = marker

    return messages


def apply_request_level_cache_control(
    api_kwargs: Dict[str, Any],
    cache_ttl: str = "5m",
) -> None:
    """Add top-level cache_control to an Anthropic API request.

    This enables automatic caching: the API places a cache breakpoint on
    the last cacheable block and moves it forward as conversations grow.
    Combined with explicit breakpoints on tools and system-reminders,
    this provides optimal caching without wasting a slot on a rolling-tail
    message marker.
    """
    marker: Dict[str, Any] = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"
    api_kwargs["cache_control"] = marker


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
