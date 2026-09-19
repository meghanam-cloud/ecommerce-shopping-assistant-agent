"""
Langfuse tracing helpers.

Provides a single ``get_tracer()`` entry point that returns either a real
Langfuse client or a no-op tracer whose objects mirror the same API
(``trace``, ``span``, ``generation``, ``update``, ``end``).  This means every
call site can unconditionally chain calls without ``if tracer:`` guards.

Usage pattern in graph.py / nodes / app.py:

    from src.observability.langfuse_tracer import get_tracer, trace_span

    tracer = get_tracer()
    trace = tracer.trace(name="shopping_assistant_turn",
                         user_id=session_id, input=question)

    # context-manager helper – span is created, yielded, ended on exit
    with trace_span(trace, "guardrail_check", input=question) as span:
        ...
        span.update(output=result)

    # for LLM calls use generation() instead of span()
    gen = trace.generation(name="llm_sql_generation", input=prompt,
                           model=model_name)
    gen.update(output=response_text)
    gen.end()

    trace.update(output=final_answer)
"""

import logging
from contextlib import contextmanager

from src.utils.config_loader import settings

logger = logging.getLogger(__name__)

_client = None


class _NoOpSpan:
    """No-op stand-in for a Langfuse span / generation / trace.

    Every method returns ``self`` (or ``None`` for ``end``) so callers can
    chain freely without null checks.  This is used both when Langfuse is
    disabled and as the object returned by ``_NoOpTracer.trace()``.
    """
    def span(self, **kwargs):
        return _NoOpSpan()

    def generation(self, **kwargs):
        return _NoOpSpan()

    def event(self, **kwargs):
        return _NoOpSpan()

    def update(self, **kwargs):
        return self

    def end(self, **kwargs):
        pass


class _NoOpTracer:
    """No-op stand-in for the Langfuse client itself."""
    def trace(self, **kwargs):
        return _NoOpSpan()


def get_tracer():
    """Return a cached Langfuse client or a no-op tracer.

    The first call initialises the client (or falls back to no-op on
    failure / when disabled).  Subsequent calls return the cached singleton.
    """
    global _client
    if _client is not None:
        return _client

    if not settings.langfuse_enabled:
        _client = _NoOpTracer()
        return _client

    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=settings.secrets.langfuse_public_key,
            secret_key=settings.secrets.langfuse_secret_key,
            host=settings.secrets.langfuse_host,
        )
    except Exception as e:
        logger.warning("Langfuse init failed, tracing disabled: %s", e)
        _client = _NoOpTracer()

    return _client


@contextmanager
def trace_span(parent, name, **kwargs):
    """Context manager that creates a span on *parent* (a trace or span),
    yields it, and ends it on exit.

    Any ``input`` or ``metadata`` kwargs are forwarded to ``parent.span()``.
    If the body raises, the span is tagged as ``ERROR`` before re-raising so
    failures are visible in the Langfuse UI.

    Usage::

        with trace_span(trace, "sql_execution", input=sql) as span:
            result = execute(sql)
            span.update(output=result)
    """
    span = parent.span(name=name, **kwargs)
    try:
        yield span
    except Exception as exc:
        try:
            span.update(level="ERROR", status_message=str(exc))
        except Exception:
            pass
        raise
    finally:
        try:
            span.end()
        except Exception:
            pass
