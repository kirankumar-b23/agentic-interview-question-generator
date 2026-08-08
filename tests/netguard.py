"""Shared state for the test-suite network guard.

This lives OUTSIDE `conftest.py` deliberately. pytest loads conftest through its own plugin machinery, so
a test doing `from tests.conftest import ...` imports a SECOND, independent module object — a different
exception class and a different ledger list, which makes the guard's own tests fail in confusing ways.
A plain importable module gives conftest and the tests the same objects.
"""
from __future__ import annotations


class NetworkAccessBlocked(RuntimeError):
    """Raised when a test attempts a real outbound connection."""


# (test nodeid, host) for every blocked attempt.
#
# Recorded rather than only raised, because every call site that leaks — `_scope_trim`,
# `_syllabus_audit`, `_same_thing_pass`, `tool_validate_relevance`, `fetch_open_web` — is deliberately
# FAIL-OPEN and swallows the exception. The first version of the guard raised and nothing else; the whole
# suite passed and reported zero leaks while calls were being attempted and discarded. The ledger is what
# actually catches them.
BLOCKED: list[tuple[str, str]] = []


def attempts_for(nodeid: str) -> list[str]:
    return [host for node, host in BLOCKED if node == nodeid]
