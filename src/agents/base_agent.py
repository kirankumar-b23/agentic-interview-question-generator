"""Base agent — each specialized agent runs its own LLM loop over a focused tool subset."""

from __future__ import annotations
import json
from typing import Callable, TYPE_CHECKING

from src.llm_client import _call_with_retry, get_client, run_model

if TYPE_CHECKING:
    from src.agent import AgentState

EmitFn = Callable[..., None]   # (step_id, status, detail, **structured_fields)


class BaseAgent:
    name: str = "base"
    display_name: str = "Base Agent"
    max_tool_calls: int = 10

    # ── Subclass API ─────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict]:
        raise NotImplementedError

    def get_tool_dispatch(self) -> dict:
        raise NotImplementedError

    def get_system_prompt(self, state: AgentState) -> str:
        raise NotImplementedError

    def get_user_prompt(self, state: AgentState) -> str:
        raise NotImplementedError

    def _should_stop_after(self, tool_name: str, tool_result: dict, state: AgentState) -> bool:
        """Override to define early-exit after a specific tool result."""
        return False

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def run(self, state: AgentState, emit: EmitFn) -> None:
        """Run this agent's LLM loop, emitting structured progress events.

        Events carry `agent`, timing and token counts alongside the prose `detail`, so the UI can
        group tool calls under the agent that made them and show real durations instead of
        re-parsing sentences.
        """
        import time
        from src.agent import _compact_tool_content, _msg_to_dict, _summarize_result

        phase_started = time.time()
        tokens_at_start = (state.api_usage.get("prompt_tokens", 0)
                           + state.api_usage.get("completion_tokens", 0))
        emit(f"phase:{self.name}", "running", f"Starting {self.display_name}...",
             agent=self.name, label=self.display_name)

        client = get_client()
        tool_schemas = self.get_tool_schemas()
        tool_dispatch = self.get_tool_dispatch()

        messages = [
            {"role": "system", "content": self.get_system_prompt(state)},
            {"role": "user", "content": self.get_user_prompt(state)},
        ]

        tool_call_count = 0
        stop_requested = False

        model = run_model(state)
        while tool_call_count < self.max_tool_calls and not stop_requested:
            try:
                # Retried on transient errors like every other LLM call. Without this a single 429
                # or 503 ended the phase outright: retrieval would return a near-empty pool, or
                # evaluation would never submit, and the run continued as if nothing happened.
                response = _call_with_retry(lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tool_schemas,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=4096,
                ))
            except Exception as exc:
                short = str(exc).split('\n')[0][:200]
                emit(f"phase:{self.name}", "error", f"{self.display_name} — API error: {short}",
                     agent=self.display_name)
                state.phase_errors.append(f"{self.display_name}: {short}")
                return

            if getattr(response, "usage", None):
                state.api_usage["llm_calls"] += 1
                state.api_usage["prompt_tokens"] += response.usage.prompt_tokens or 0
                state.api_usage["completion_tokens"] += response.usage.completion_tokens or 0

            msg = response.choices[0].message
            messages.append(_msg_to_dict(msg))

            if not msg.tool_calls:
                break

            for tool_call in msg.tool_calls:
                tool_call_count += 1
                name = tool_call.function.name

                try:
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                    emit(name, "warning", f"Bad JSON args for {name}", agent=self.name)

                emit(name, "running", f"{self.display_name}: calling {name}...",
                     agent=self.name, label=self.display_name)
                tool_started = time.time()
                tool_tokens_at_start = (state.api_usage.get("prompt_tokens", 0)
                                        + state.api_usage.get("completion_tokens", 0))

                try:
                    handler = tool_dispatch.get(name)
                    tool_result = handler(state, **args) if handler else {"error": f"Unknown tool: {name}"}
                except Exception as exc:
                    tool_result = {"error": str(exc)}

                state.tool_log.append({
                    "agent": self.name,
                    "tool": name,
                    "args_keys": list(args.keys()),
                    "has_error": "error" in tool_result,
                })

                if name == "deduplicate_questions" and "removed" in tool_result:
                    state.dedup_removed += tool_result.get("removed", 0)

                summary = _summarize_result(name, tool_result)
                # Per-step token cost: a tool that makes its own LLM calls (validate_relevance) is
                # where the spend actually goes, so attributing tokens only per phase hid it.
                tool_tokens = ((state.api_usage.get("prompt_tokens", 0)
                                + state.api_usage.get("completion_tokens", 0)) - tool_tokens_at_start)
                emit(name, "done" if "error" not in tool_result else "error", summary,
                     agent=self.name, label=self.display_name,
                     duration_ms=int((time.time() - tool_started) * 1000),
                     tokens=tool_tokens or None,
                     pool=len(state.questions))

                result_str = json.dumps(tool_result, default=str)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": _compact_tool_content(result_str),
                })

                if self._should_stop_after(name, tool_result, state):
                    stop_requested = True
                    break

        q_count = len(state.questions) + len(state.coding_questions)
        tokens_used = ((state.api_usage.get("prompt_tokens", 0)
                        + state.api_usage.get("completion_tokens", 0)) - tokens_at_start)
        emit(f"phase:{self.name}", "done",
             f"{self.display_name} complete — {q_count} questions, {tool_call_count} calls",
             agent=self.name, label=self.display_name,
             duration_ms=int((time.time() - phase_started) * 1000),
             tool_calls=tool_call_count, questions=q_count, tokens=tokens_used)
