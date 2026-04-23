"""Google Gemini provider for agent_search."""

import os
import time
import uuid


def _tools_for_gemini(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool defs to Gemini function declarations.

    Gemini generates MALFORMED_FUNCTION_CALL when a tool declaration has an
    empty-properties schema (e.g. build/run_tests/benchmark).  Omit the
    parameters key entirely for no-argument tools so Gemini knows they take none.
    """
    decls = []
    for t in tools:
        decl = {"name": t["name"], "description": t["description"]}
        schema = t.get("input_schema", {})
        if schema.get("properties"):
            decl["parameters"] = schema
        decls.append(decl)
    return [{"function_declarations": decls}]


def _messages_to_contents(messages: list[dict]) -> list[dict]:
    """Translate shared message history to Gemini contents format."""
    contents = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        raw = msg["content"]
        if isinstance(raw, str):
            contents.append({"role": role, "parts": [{"text": raw}]})
            continue
        parts: list[dict] = []
        for block in raw:
            btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if btype == "text":
                text = block["text"] if isinstance(block, dict) else block.text
                if text:
                    parts.append({"text": text})
            elif btype == "tool_use":
                name = block["name"] if isinstance(block, dict) else block.name
                inp  = block["input"] if isinstance(block, dict) else block.input
                fc_dict: dict = {"name": name, "args": inp or {}}
                # Echo thought_signature back — required by Gemini thinking models.
                sig = block.get("thought_signature") if isinstance(block, dict) else getattr(block, "thought_signature", None)
                if sig is not None:
                    fc_dict["thought_signature"] = sig
                parts.append({"function_call": fc_dict})
            elif btype == "tool_result":
                fn_name = block.get("tool_name", block.get("tool_use_id", "tool"))
                result  = block.get("content", "")
                parts.append({"function_response": {"name": fn_name, "response": {"result": result}}})
            elif btype == "gemini_thought":
                text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                parts.append({"thought": True, "text": text})
            # Anthropic-style thinking blocks are not forwarded to Gemini
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents


def call(
    messages: list[dict],
    system: str,
    model: str,
    max_tokens: int,
    tools: list[dict],
) -> tuple[list[dict], str]:
    """Call Gemini and return (content_dicts, stop_reason)."""
    try:
        from google import genai as _genai
        from google.genai import types as _gtypes
    except ImportError:
        raise RuntimeError("google-genai not installed — run: pip install google-genai")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")

    gclient = _genai.Client(api_key=api_key)

    max_retries = 5
    base_delay = 30.0
    for attempt in range(max_retries):
        try:
            response = gclient.models.generate_content(
                model=model,
                contents=_messages_to_contents(messages),
                config=_gtypes.GenerateContentConfig(
                    system_instruction=system,
                    tools=_tools_for_gemini(tools),
                    max_output_tokens=max_tokens,
                ),
            )
            break
        except Exception as _e:
            err_str = str(_e)
            if "429" not in err_str and "RESOURCE_EXHAUSTED" not in err_str:
                raise
            if "PerDay" in err_str or "per_day" in err_str.lower():
                raise RuntimeError(
                    "Gemini daily quota exhausted. "
                    "Upgrade your plan or wait until quota resets (midnight Pacific).\n"
                    f"  Original error: {_e}"
                ) from None
            if attempt == max_retries - 1:
                raise
            import re as _re
            delay_match = _re.search(r"retryDelay.*?(\d+)s", err_str)
            delay = float(delay_match.group(1)) if delay_match else base_delay * (2 ** attempt)
            print(f"\n[Gemini] Rate limited (429), retrying in {delay:.0f}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(delay)

    content_dicts: list[dict] = []
    stop_reason = "end_turn"
    candidate = response.candidates[0] if response.candidates else None

    if candidate is None:
        fb = getattr(response, "prompt_feedback", None)
        raise RuntimeError(
            f"Gemini returned no candidates — prompt may have been blocked.\n"
            f"  prompt_feedback: {fb}"
        )

    if candidate.content is None:
        finish = str(getattr(candidate, "finish_reason", "unknown"))
        if "MALFORMED_FUNCTION_CALL" in finish:
            print(f"\n[Gemini] Warning: malformed function call ({finish}), nudging model to retry.")
            return [{"type": "text", "text": "My previous tool call was malformed. I will retry with correct syntax."}], "tool_use"
        raise RuntimeError(
            f"Gemini candidate has no content (finish_reason={finish!r}).\n"
            f"  The response was likely filtered. Try a different model or simplify the system prompt."
        )

    for part in (candidate.content.parts or []):
        if getattr(part, "thought", False):
            content_dicts.append({"type": "gemini_thought", "text": getattr(part, "text", "") or ""})
            continue
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None):
            entry: dict = {
                "type":  "tool_use",
                "id":    f"gemini-{uuid.uuid4().hex[:8]}",
                "name":  fc.name,
                "input": dict(fc.args) if fc.args else {},
            }
            sig = getattr(fc, "thought_signature", None)
            if sig is not None:
                entry["thought_signature"] = sig
            content_dicts.append(entry)
            stop_reason = "tool_use"
        elif getattr(part, "text", None):
            content_dicts.append({"type": "text", "text": part.text})

    finish = str(getattr(candidate, "finish_reason", ""))
    if "MAX_TOKENS" in finish:
        stop_reason = "max_tokens"
    elif "STOP" in finish and stop_reason != "tool_use":
        stop_reason = "end_turn"

    return content_dicts, stop_reason
