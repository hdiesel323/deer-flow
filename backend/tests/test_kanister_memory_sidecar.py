import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware
from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
from deerflow.config.memory_config import MemoryConfig


class _Response:
    def __init__(self, payload: dict, status: int = 200):
        self.payload = payload
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _runtime(**context):
    return SimpleNamespace(context=context)


def _app_config(**memory_overrides):
    return SimpleNamespace(memory=MemoryConfig(**memory_overrides))


def _request_body(request) -> dict:
    return json.loads(request.data.decode())


def test_dynamic_context_posts_recall_request_shape_and_injects_labels(monkeypatch):
    from deerflow.agents.memory import kanister_sidecar

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = _request_body(request)
        return _Response(
            {
                "items": [
                    {
                        "content": "Prefer concise execution summaries.",
                        "freshness": "fresh",
                        "provenance": {"source": "kanister:test-record"},
                    },
                    {
                        "text": "Old project name was RetriEval.",
                        "stale": True,
                        "provenance": "legacy-note",
                    },
                ]
            }
        )

    monkeypatch.setattr(kanister_sidecar.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("deerflow.agents.lead_agent.prompt._get_memory_context", lambda *args, **kwargs: "")
    monkeypatch.setattr("deerflow.agents.middlewares.dynamic_context_middleware.datetime", MagicMock())

    import deerflow.agents.middlewares.dynamic_context_middleware as dynamic_context_module

    dynamic_context_module.datetime.now.return_value.strftime.return_value = "2026-05-08, Friday"

    mw = DynamicContextMiddleware(app_config=_app_config(kanister={"enabled": True, "timeout_seconds": 0.25, "recall_budget": 3}))
    result = mw.before_agent(
        {"messages": [HumanMessage(content="Use my OPENAI_API_KEY=sk-test-secret for this?", id="msg-1")], "thread_data": {"workspace_path": "/tmp/workspace"}},
        _runtime(thread_id="thread-1", run_id="run-1", user_id="alice", workspace_id="workspace-1", task_id="RED-3060", memory_scopes=["user", "task"]),
    )

    assert captured["url"] == "http://127.0.0.1:17890/v1/recall"
    assert captured["timeout"] == 0.25
    assert captured["body"]["actor"] == {"user_id": "alice"}
    assert captured["body"]["harness"]["name"] == "deerflow"
    assert captured["body"]["workspace"]["id"] == "workspace-1"
    assert captured["body"]["task"]["id"] == "RED-3060"
    assert captured["body"]["session"] == {"thread_id": "thread-1", "run_id": "run-1", "session_id": "thread-1"}
    assert captured["body"]["scopes"] == ["user", "task"]
    assert captured["body"]["budget"] == {"items": 3}
    assert "sk-test-secret" not in json.dumps(captured["body"])

    assert "kanister-memory" not in result["messages"][0].content
    memory_context = result["messages"][1].content
    assert '<kanister-memory status="available">' in memory_context
    assert "[fresh] [provenance: kanister:test-record] Prefer concise execution summaries." in memory_context
    assert "[stale] [provenance: legacy-note] Old project name was RetriEval." in memory_context


def test_dynamic_context_marks_partial_and_unavailable(monkeypatch):
    from deerflow.agents.memory import kanister_sidecar

    calls = []

    def partial_urlopen(request, timeout):
        calls.append(request.full_url)
        return _Response({"partial": True, "items": [{"content": "Partial fact", "provenance": "sidecar"}]})

    monkeypatch.setattr(kanister_sidecar.request, "urlopen", partial_urlopen)
    monkeypatch.setattr("deerflow.agents.lead_agent.prompt._get_memory_context", lambda *args, **kwargs: "")
    monkeypatch.setattr("deerflow.agents.middlewares.dynamic_context_middleware.datetime", MagicMock())

    import deerflow.agents.middlewares.dynamic_context_middleware as dynamic_context_module

    dynamic_context_module.datetime.now.return_value.strftime.return_value = "2026-05-08, Friday"

    mw = DynamicContextMiddleware(app_config=_app_config(kanister={"enabled": True}))
    partial = mw.before_agent({"messages": [HumanMessage(content="hi", id="msg-1")]}, _runtime(thread_id="thread-1", run_id="run-1"))["messages"][1].content
    assert '<kanister-memory status="partial">' in partial

    def failing_urlopen(request, timeout):
        raise OSError("sidecar offline")

    monkeypatch.setattr(kanister_sidecar.request, "urlopen", failing_urlopen)
    mw = DynamicContextMiddleware(app_config=_app_config(kanister={"enabled": True}))
    unavailable = mw.before_agent({"messages": [HumanMessage(content="hi", id="msg-2")]}, _runtime(thread_id="thread-1", run_id="run-2"))["messages"][1].content
    assert '<kanister-memory status="unavailable">' in unavailable
    assert "sidecar offline" in unavailable


def test_dynamic_context_disabled_modes_do_not_call_sidecar(monkeypatch):
    from deerflow.agents.memory import kanister_sidecar

    def fail_urlopen(request, timeout):
        raise AssertionError("sidecar must not be called when memory or injection is disabled")

    monkeypatch.setattr(kanister_sidecar.request, "urlopen", fail_urlopen)
    monkeypatch.setattr("deerflow.agents.lead_agent.prompt._get_memory_context", lambda *args, **kwargs: "")
    monkeypatch.setattr("deerflow.agents.middlewares.dynamic_context_middleware.datetime", MagicMock())

    import deerflow.agents.middlewares.dynamic_context_middleware as dynamic_context_module

    dynamic_context_module.datetime.now.return_value.strftime.return_value = "2026-05-08, Friday"

    disabled_memory = DynamicContextMiddleware(app_config=_app_config(enabled=False, kanister={"enabled": True}))
    disabled_injection = DynamicContextMiddleware(app_config=_app_config(injection_enabled=False, kanister={"enabled": True}))
    date_only_reminder = "<system-reminder>\n<current_date>2026-05-08, Friday</current_date>\n</system-reminder>"

    assert disabled_memory.before_agent({"messages": [HumanMessage(content="hi", id="msg-1")]}, _runtime(thread_id="thread-1"))["messages"][0].content == date_only_reminder
    assert disabled_injection.before_agent({"messages": [HumanMessage(content="hi", id="msg-2")]}, _runtime(thread_id="thread-1"))["messages"][0].content == date_only_reminder


def test_dynamic_context_frozen_reminder_does_not_recall_again(monkeypatch):
    from deerflow.agents.memory import kanister_sidecar

    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return _Response({"items": [{"content": "fact", "provenance": "sidecar"}]})

    monkeypatch.setattr(kanister_sidecar.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("deerflow.agents.lead_agent.prompt._get_memory_context", lambda *args, **kwargs: "")
    monkeypatch.setattr("deerflow.agents.middlewares.dynamic_context_middleware.datetime", MagicMock())

    import deerflow.agents.middlewares.dynamic_context_middleware as dynamic_context_module

    dynamic_context_module.datetime.now.return_value.strftime.return_value = "2026-05-08, Friday"

    mw = DynamicContextMiddleware(app_config=_app_config(kanister={"enabled": True}))
    first = mw.before_agent({"messages": [HumanMessage(content="hi", id="msg-1")]}, _runtime(thread_id="thread-1", run_id="run-1"))
    second = mw.before_agent({"messages": first["messages"] + [AIMessage(content="hello"), HumanMessage(content="next", id="msg-2")]}, _runtime(thread_id="thread-1", run_id="run-2"))

    assert len(calls) == 1
    assert second is None


def test_memory_middleware_preserves_queue_and_posts_idempotent_write(monkeypatch):
    from deerflow.agents.memory import kanister_sidecar

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = _request_body(request)
        return _Response({"ok": True})

    manager = MagicMock()
    monkeypatch.setattr(kanister_sidecar.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("deerflow.agents.middlewares.memory_middleware.get_memory_manager", lambda: manager)

    mw = MemoryMiddleware(memory_config=MemoryConfig(kanister={"enabled": True}))
    result = mw.after_agent(
        {"messages": [HumanMessage(content="my password = swordfish", id="h1"), AIMessage(content="Done. Token Bearer abc123SECRET", id="a1")]},
        _runtime(thread_id="thread-1", run_id="run-1", user_id="alice", workspace_id="workspace-1"),
    )

    assert result is None
    manager.add.assert_called_once()
    assert captured["url"] == "http://127.0.0.1:17890/v1/write"
    event = captured["body"]
    assert event["schema"] == "SessionEventV1"
    assert event["type"] == "outcome"
    assert event["idempotency_key"]
    assert event["actor"] == {"user_id": "alice"}
    assert event["session"]["thread_id"] == "thread-1"
    assert event["session"]["run_id"] == "run-1"
    assert "swordfish" not in json.dumps(event)
    assert "abc123SECRET" not in json.dumps(event)


def test_memory_middleware_write_outage_is_non_blocking(monkeypatch):
    from deerflow.agents.memory import kanister_sidecar

    def failing_urlopen(request, timeout):
        raise OSError("sidecar offline")

    manager = MagicMock()
    monkeypatch.setattr(kanister_sidecar.request, "urlopen", failing_urlopen)
    monkeypatch.setattr("deerflow.agents.middlewares.memory_middleware.get_memory_manager", lambda: manager)

    mw = MemoryMiddleware(memory_config=MemoryConfig(kanister={"enabled": True}))
    result = mw.after_agent(
        {"messages": [HumanMessage(content="hello", id="h1"), AIMessage(content="done", id="a1")]},
        _runtime(thread_id="thread-1", run_id="run-1", user_id="alice"),
    )

    assert result is None
    manager.add.assert_called_once()
