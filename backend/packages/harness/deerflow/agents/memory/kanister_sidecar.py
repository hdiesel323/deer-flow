"""Best-effort stdlib HTTP bridge to the local Kanister memory sidecar."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from html import escape
from typing import Any
from urllib import request

from langchain_core.messages import BaseMessage

from deerflow.config.memory_config import MemoryConfig
from deerflow.runtime.user_context import DEFAULT_USER_ID, resolve_runtime_user_id

logger = logging.getLogger(__name__)

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(password|passwd|pwd|api[_-]?key|access[_-]?token|refresh[_-]?token|secret|secret[_-]?key)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
)


def redact_credentials(value: Any) -> str:
    """Return a string safe to send to the sidecar without obvious credentials."""
    text = value if isinstance(value, str) else str(value)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(_redact_match, text)
    return text


def _redact_match(match: re.Match[str]) -> str:
    matched = match.group(0)
    if matched.lower().startswith("bearer "):
        return "Bearer [REDACTED]"
    if matched.startswith("sk-"):
        return "sk-[REDACTED]"
    if "=" in matched:
        return f"{matched.split('=', 1)[0].strip()}=[REDACTED]"
    if ":" in matched:
        return f"{matched.split(':', 1)[0].strip()}: [REDACTED]"
    return "[REDACTED]"


def recall_reminder(memory_config: MemoryConfig, *, state: dict[str, Any], runtime: Any, agent_name: str | None, query: Any) -> str | None:
    """Fetch and format Kanister recall context for the frozen system reminder."""
    if not _kanister_enabled_for_recall(memory_config):
        return None

    payload = _build_recall_payload(memory_config, state=state, runtime=runtime, agent_name=agent_name, query=query)
    try:
        response = _post_json(memory_config.kanister.base_url, "/v1/recall", payload, timeout=memory_config.kanister.timeout_seconds)
    except Exception as exc:
        logger.info("Kanister recall unavailable: %s", exc)
        return _format_unavailable(exc)
    return _format_recall_response(response)


def emit_session_outcome(memory_config: MemoryConfig, *, state: dict[str, Any], runtime: Any, agent_name: str | None, messages: list[BaseMessage]) -> bool:
    """Emit a best-effort SessionEventV1/outcome event without blocking execution."""
    if not _kanister_enabled(memory_config):
        return False

    payload = _build_outcome_payload(state=state, runtime=runtime, agent_name=agent_name, messages=messages)
    try:
        _post_json(memory_config.kanister.base_url, "/v1/write", payload, timeout=memory_config.kanister.timeout_seconds)
    except Exception as exc:
        logger.info("Kanister outcome write skipped: %s", exc)
        return False
    return True


def _kanister_enabled(memory_config: MemoryConfig) -> bool:
    return bool(memory_config.enabled and memory_config.kanister.enabled)


def _kanister_enabled_for_recall(memory_config: MemoryConfig) -> bool:
    return bool(_kanister_enabled(memory_config) and memory_config.injection_enabled)


def _build_recall_payload(memory_config: MemoryConfig, *, state: dict[str, Any], runtime: Any, agent_name: str | None, query: Any) -> dict[str, Any]:
    context = _runtime_context(runtime)
    thread_id = _context_value(context, "thread_id")
    run_id = _context_value(context, "run_id")
    session_id = _context_value(context, "session_id") or thread_id
    return {
        "actor": {"user_id": _actor_id(runtime)},
        "harness": {"name": "deerflow", "agent": agent_name or "lead_agent"},
        "workspace": _workspace_payload(state, context),
        "task": {"id": _context_value(context, "task_id")},
        "session": {"thread_id": thread_id, "run_id": run_id, "session_id": session_id},
        "scopes": _scopes(memory_config, context),
        "query": redact_credentials(_message_content_to_text(query)),
        "budget": {"items": memory_config.kanister.recall_budget},
    }


def _build_outcome_payload(*, state: dict[str, Any], runtime: Any, agent_name: str | None, messages: list[BaseMessage]) -> dict[str, Any]:
    context = _runtime_context(runtime)
    actor = {"user_id": _actor_id(runtime)}
    session = {
        "thread_id": _context_value(context, "thread_id"),
        "run_id": _context_value(context, "run_id"),
        "session_id": _context_value(context, "session_id") or _context_value(context, "thread_id"),
    }
    event_messages = [_message_payload(message) for message in messages]
    seed = json.dumps(
        {
            "actor": actor,
            "session": session,
            "agent": agent_name or "lead_agent",
            "last_message": event_messages[-1] if event_messages else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "schema": "SessionEventV1",
        "type": "outcome",
        "idempotency_key": hashlib.sha256(seed.encode()).hexdigest(),
        "actor": actor,
        "harness": {"name": "deerflow", "agent": agent_name or "lead_agent"},
        "workspace": _workspace_payload(state, context),
        "task": {"id": _context_value(context, "task_id")},
        "session": session,
        "outcome": {"messages": event_messages},
    }


def _post_json(base_url: str, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        if not raw:
            return {}
        return json.loads(raw.decode())


def _format_recall_response(response: dict[str, Any]) -> str:
    items = _normalize_items(response)
    status = "partial" if response.get("partial") is True else "available"
    if not items:
        return f'<kanister-memory status="{status}">\nNo Kanister recall items returned.\n</kanister-memory>'
    return "\n".join(
        [
            f'<kanister-memory status="{status}">',
            *items,
            "</kanister-memory>",
        ]
    )


def _format_unavailable(exc: Exception) -> str:
    detail = redact_credentials(str(exc)) or exc.__class__.__name__
    return "\n".join(
        [
            '<kanister-memory status="unavailable">',
            f"Kanister memory unavailable: {escape(detail)}",
            "</kanister-memory>",
        ]
    )


def _normalize_items(response: dict[str, Any]) -> list[str]:
    raw_items = response.get("items")
    if not isinstance(raw_items, list):
        raw_items = response.get("results") if isinstance(response.get("results"), list) else []
    items: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            content = str(raw_item).strip()
            freshness = "freshness: unknown"
            provenance = "unknown"
        else:
            content = str(raw_item.get("content") or raw_item.get("text") or raw_item.get("summary") or "").strip()
            freshness = _freshness_label(raw_item)
            provenance = _provenance_label(raw_item.get("provenance"))
        if not content:
            continue
        items.append(f"[{freshness}] [provenance: {escape(provenance)}] {escape(redact_credentials(content))}")
    return items


def _freshness_label(item: dict[str, Any]) -> str:
    if item.get("stale") is True:
        return "stale"
    freshness = item.get("freshness")
    if isinstance(freshness, str) and freshness.strip():
        return freshness.strip()
    return "freshness: unknown"


def _provenance_label(provenance: Any) -> str:
    if isinstance(provenance, dict):
        for key in ("source", "id", "uri", "url"):
            value = provenance.get(key)
            if value:
                return str(value)
        return "unknown"
    if provenance:
        return str(provenance)
    return "unknown"


def _message_payload(message: BaseMessage) -> dict[str, str | None]:
    return {
        "id": getattr(message, "id", None),
        "role": getattr(message, "type", None),
        "content": redact_credentials(_message_content_to_text(getattr(message, "content", ""))),
    }


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    return str(content)


def _runtime_context(runtime: Any) -> dict[str, Any]:
    context = getattr(runtime, "context", None)
    return context if isinstance(context, dict) else {}


def _context_value(context: dict[str, Any], key: str) -> str | None:
    value = context.get(key)
    if value is None:
        return None
    return str(value)


def _actor_id(runtime: Any) -> str:
    try:
        return resolve_runtime_user_id(runtime)
    except Exception:
        return DEFAULT_USER_ID


def _workspace_payload(state: dict[str, Any], context: dict[str, Any]) -> dict[str, str | None]:
    thread_data = state.get("thread_data")
    workspace_path = thread_data.get("workspace_path") if isinstance(thread_data, dict) else None
    return {
        "id": _context_value(context, "workspace_id"),
        "path": redact_credentials(workspace_path) if workspace_path else None,
    }


def _scopes(memory_config: MemoryConfig, context: dict[str, Any]) -> list[str]:
    runtime_scopes = context.get("memory_scopes")
    if isinstance(runtime_scopes, list):
        return [str(scope) for scope in runtime_scopes]
    return list(memory_config.kanister.scopes)
