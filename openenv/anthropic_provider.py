"""Anthropic/Claude provider for agent_search."""

import time
import anthropic


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Strip provider-specific fields that Anthropic rejects (e.g. tool_name in tool_result)."""
    clean = []
    for msg in messages:
        content = msg["content"]
        if isinstance(content, list):
            new_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    block = {k: v for k, v in block.items() if k != "tool_name"}
                new_content.append(block)
            clean.append({**msg, "content": new_content})
        else:
            clean.append(msg)
    return clean


def call(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int,
    tools: list[dict],
) -> tuple[list[dict], str]:
    """Call Anthropic and return (content_dicts, stop_reason)."""
    client = anthropic.Anthropic()

    for attempt in range(5):
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                system=system,
                tools=tools,
                messages=_sanitize_messages(messages),
            ) as stream:
                response = stream.get_final_message()
            break
        except anthropic.RateLimitError as e:
            wait = 60 * (attempt + 1)
            print(f"\n[Rate limit] Waiting {wait}s before retry ({attempt+1}/5)… ({e})")
            time.sleep(wait)
    else:
        raise RuntimeError("Rate limit retries exhausted")

    content_dicts: list[dict] = []
    for block in response.content:
        if block.type == "thinking":
            entry: dict = {"type": "thinking", "thinking": getattr(block, "thinking", "")}
            sig = getattr(block, "signature", None)
            if sig:
                entry["signature"] = sig
            content_dicts.append(entry)
        elif block.type == "text":
            content_dicts.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content_dicts.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })

    return content_dicts, response.stop_reason
