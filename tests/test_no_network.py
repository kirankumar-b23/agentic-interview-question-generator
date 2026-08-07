"""The suite must cost no API credit, and this proves the guard enforcing that is armed.

Without these, a refactor could disable the `tests/conftest.py` guard and every other test would still
pass — which is precisely the failure this whole guard exists to stop. The suite ran green for months
while spending money on Tavily and OpenRouter.

The second test is the important one. The FIRST version of the guard only raised, and every leaking call
site (`_scope_trim`, `_syllabus_audit`, `_same_thing_pass`, `tool_validate_relevance`, `fetch_open_web`)
is deliberately fail-open — so it caught the exception and carried on. The whole suite passed and
reported zero leaks while calls were being attempted and swallowed. The guard therefore RECORDS attempts
and fails the offending test in teardown; a raise alone is not enough.
"""
import socket

import pytest


class TestTheGuardIsArmed:

    @pytest.fixture(autouse=True)
    def _forget_our_deliberate_attempts(self, request):
        """This file is the one place that attempts connections ON PURPOSE.

        The conftest guard fails any test that recorded an attempt, so these tests must clear their own
        entries. Module-level fixtures finalise BEFORE conftest's autouse one, so this runs first.
        """
        yield
        from tests import netguard
        netguard.BLOCKED[:] = [t for t in netguard.BLOCKED if t[0] != request.node.nodeid]

    def test_a_raw_outbound_connection_is_blocked_and_names_the_host(self):
        from tests.netguard import NetworkAccessBlocked

        s = socket.socket()
        s.settimeout(3)
        with pytest.raises(NetworkAccessBlocked) as exc:
            s.connect(("openrouter.ai", 443))
        assert "openrouter.ai" in str(exc.value), "the error must name the host, or triage is guesswork"

    def test_an_http_client_is_blocked_too(self):
        """The leaks were all httpx/openai-SDK calls, not raw sockets."""
        import httpx

        from tests.netguard import NetworkAccessBlocked

        with pytest.raises((NetworkAccessBlocked, httpx.ConnectError, httpx.TransportError)):
            httpx.get("https://openrouter.ai/api/v1/key", timeout=5)

    def test_loopback_is_still_allowed(self):
        """Starlette's TestClient is in-process, and local fixtures must keep working."""
        s = socket.socket()
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", 1))
        except ConnectionRefusedError:
            pass                        # refused by the OS = the guard let it through, which is correct
        except Exception as exc:        # noqa: BLE001 — any guard error here is the failure
            pytest.fail(f"loopback must not be blocked: {exc!r}")

    def test_attempts_are_recorded_not_merely_raised(self):
        """A fail-open caller swallows the raise, so the ledger is what actually catches a leak."""
        from tests import netguard

        before = len(netguard.BLOCKED)
        try:                            # exactly what a fail-open call site does
            s = socket.socket()
            s.settimeout(2)
            s.connect(("example.invalid", 443))
        except Exception:               # noqa: BLE001 — deliberately swallowed, as production code does
            pass
        assert len(netguard.BLOCKED) > before, (
            "a swallowed attempt must still be recorded, or fail-open code paths leak silently")
        # Keep this test's own deliberate attempt out of the teardown check.
        netguard.BLOCKED[:] = [t for t in netguard.BLOCKED if t[1] != "example.invalid"]
