"""Make the project root importable, and make "no network" an ENFORCED guarantee.

WHY THE NETWORK GUARD EXISTS
----------------------------
CLAUDE.md has always claimed this suite needs "no LLM or network required". Nothing enforced it, and it
was false in three places — so every `pytest tests/` run quietly spent real money:

  * `tests/test_pipeline_integration.py` runs the pipeline 21 times and stubbed
    `tool_search_web_questions` but NOT `fetch_open_web`, the last-resort tier. That was harmless while
    the tier only fired below MIN_QUESTIONS; once the zero-tool-representation trigger began firing at any
    count it went live on nearly every run — up to 4 Tavily searches each, so ~84 per suite run. It
    exhausted the Tavily plan, and the giveaway was a search for "Observation interview questions".
  * `test_run_defects.py` called `tool_submit_question_set` unstubbed, which makes three OpenRouter calls
    (`_scope_trim`, `_syllabus_audit`, `_same_thing_pass`). That one test took 4.9s of pure round-trips.
  * `test_api.py`'s `/api/usage` case reached `get_credit_balance()`.

Each was invisible because a live call that succeeds looks exactly like a stub that works. The guard makes
the failure mode loud instead: any outbound connection raises and NAMES THE HOST, so the next leak is
found in one test run rather than on an exhausted quota.

Loopback stays open so Starlette's in-process `TestClient` and any local fixture keep working, and unix
sockets are untouched.

If a test genuinely needs the network, mark it `@pytest.mark.allow_network` — deliberately applied to
nothing today, so an exception is explicit and greppable rather than a silent regression.
"""
import os
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.netguard import BLOCKED, NetworkAccessBlocked, attempts_for  # noqa: E402

# Loopback only. Anything else is a real egress.
_ALLOWED_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_network: permit real outbound connections in this test (currently unused)")


def _host_of(address):
    if isinstance(address, (tuple, list)) and address:
        return address[0]
    return address                      # AF_UNIX path, or an already-plain host


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Block outbound connections, record every attempt, and fail the test that made one."""
    if request.node.get_closest_marker("allow_network"):
        return

    # sentence-transformers can try to reach the HF hub on load even when the model is cached in
    # .cache/. Set offline rather than weakening the guard — several tests measure embedding behaviour
    # and must keep working.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    real_connect = socket.socket.connect
    nodeid = request.node.nodeid

    def guarded(self, address, *args, **kwargs):
        host = _host_of(address)
        if isinstance(host, str) and not host.startswith("/") and host not in _ALLOWED_HOSTS:
            BLOCKED.append((nodeid, host))
            raise NetworkAccessBlocked(
                f"Test attempted a real network connection to {host!r} ({nodeid}). "
                f"Stub it — the suite must cost no API credit. See tests/conftest.py.")
        return real_connect(self, address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    yield
    mine = attempts_for(nodeid)
    if mine:
        pytest.fail(f"attempted {len(mine)} real network connection(s) to {sorted(set(mine))}. "
                    f"The call was swallowed by a fail-open handler, so the assertions still passed — "
                    f"but a working key would have spent credit here.", pytrace=False)


@pytest.fixture
def allow_real_network(monkeypatch):
    """Escape hatch for a test that must reach out; nothing uses it."""
    monkeypatch.undo()
    yield
