import re
import json
import time
from openai import OpenAI
from src.config import (LLM_MODEL, LLM_TIMEOUT_SECONDS, OPENROUTER_API_KEY,
                        OPENROUTER_BASE_URL)

# ── OpenRouter credit balance (cached) ────────────────────────────────────────
_credit_cache: dict = {"at": 0.0, "value": None}
_CREDIT_TTL = 60  # seconds


def get_credit_balance() -> dict | None:
    """Return OpenRouter balance for the current key, cached ~60s.

    Shape: {remaining, scope, account_remaining, key_limit, key_remaining}.
    `remaining` is the binding spendable balance — the per-key limit if the key
    is capped, else the account balance. Returns None on failure.
    """
    now = time.time()
    if _credit_cache["value"] is not None and now - _credit_cache["at"] < _CREDIT_TTL:
        return _credit_cache["value"]
    if not OPENROUTER_API_KEY:
        return None

    import requests
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    result: dict = {}
    try:
        # Account-level balance
        c = requests.get(f"{OPENROUTER_BASE_URL}/credits", headers=headers, timeout=10)
        if c.ok:
            d = c.json().get("data", {})
            total = d.get("total_credits")
            used = d.get("total_usage")
            if total is not None and used is not None:
                result["account_remaining"] = round(total - used, 2)

        # Per-key limit (the binding cap, if any)
        k = requests.get(f"{OPENROUTER_BASE_URL}/auth/key", headers=headers, timeout=10)
        if k.ok:
            d = k.json().get("data", {})
            result["key_limit"] = d.get("limit")
            rem = d.get("limit_remaining")
            if rem is not None:
                result["key_remaining"] = round(rem, 2)
    except Exception:
        return _credit_cache["value"]  # serve last good value if any

    if not result:
        return None

    if result.get("key_remaining") is not None:
        result["remaining"] = result["key_remaining"]
        result["scope"] = "key"
    elif result.get("account_remaining") is not None:
        result["remaining"] = result["account_remaining"]
        result["scope"] = "account"

    _credit_cache.update(at=now, value=result)
    return result

_TRANSIENT_SIGNALS = ('429', '500', '502', '503', 'Connection', 'Timeout', 'timeout', 'rate limit')

# Model most recently chosen in the UI. This is a DISPLAY default only — it seeds the picker and
# `/api/meta`. It must never decide which model a run uses: two browser tabs generating at once would
# retarget each other's in-flight calls mid-run, and the run's own cost accounting (which stamps the
# model into api_usage) would then price its tokens at the wrong rate.
#
# The model a run actually uses comes from its own GenerationConfig, threaded through AgentState —
# see `run_model()`.
_ui_model_default: str | None = None


def set_active_model(model: str | None):
    """Remember the UI's model choice for display/defaulting purposes."""
    global _ui_model_default
    _ui_model_default = model.strip() if model and model.strip() else None


def get_active_model() -> str:
    """The UI's currently-selected model (display default), NOT necessarily any run's model."""
    return _ui_model_default or LLM_MODEL


def run_model(state) -> str:
    """The model THIS run must use: its own config, else the configured default.

    Every LLM call made on behalf of a run has to go through this. The agent tool-loops previously
    hardcoded the `LLM_MODEL` constant, so the UI picker silently did nothing for the bulk of the
    work while cost estimates were computed at the selected model's price — picking Opus inflated
    the reported cost ~15x and changed nothing about the output.
    """
    cfg = getattr(state, "config", None)
    chosen = getattr(cfg, "model", None) if cfg is not None else None
    return (chosen or "").strip() or LLM_MODEL


def _call_with_retry(fn, max_retries: int = 3):
    """Retry fn() on transient API errors with exponential backoff (1s, 2s, 4s)."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            err = str(exc)
            is_transient = any(sig in err for sig in _TRANSIENT_SIGNALS)
            if not is_transient or attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            # Without an explicit timeout the SDK waits 600s per request, so one wedged call could
            # hold a run (and its SSE stream) open for ten minutes.
            timeout=LLM_TIMEOUT_SECONDS,
            # The SDK retries twice by default. Combined with our own _call_with_retry(3) that is up
            # to 9 requests and ~7s of sleep per logical call, with the backoff applied at the wrong
            # layer. Retries are handled in _call_with_retry, which knows what's transient.
            max_retries=0,
        )
    return _client


def _extract_json(text: str) -> dict | None:
    """Parse JSON out of a model reply, tolerating markdown fences and surrounding prose.

    Returns None when nothing parses — distinct from `{}`, which is a valid empty object.
    """
    if not text:
        return None
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    on_usage=None,
) -> str:
    """Simple chat completion. Returns the text response."""
    client = get_client()
    response = _call_with_retry(lambda: client.chat.completions.create(
        model=model or get_active_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    ))
    if on_usage and getattr(response, "usage", None):
        on_usage(response.usage)
    return response.choices[0].message.content or ""


class JSONResponseError(RuntimeError):
    """The model's reply could not be parsed as JSON.

    Raised instead of returning `{}` so callers can tell "the model said nothing usable" from
    "the model returned an empty object". Returning a bare `{}` made every failure look like a
    valid-but-empty answer, which is how a truncated critique response became an automatic
    quality-gate PASS (`.get("pass", True)`).
    """

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


# Providers differ on whether they accept response_format={"type":"json_object"}; Anthropic models
# via OpenRouter generally reject it. Remember the answer per model so we stop paying for a request
# that always fails on the first try (the old code retried the same doomed call on every JSON call).
_supports_json_mode: dict[str, bool] = {}


def chat_completion_json(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    on_usage=None,
) -> dict:
    """Chat completion returning parsed JSON.

    Raises `JSONResponseError` when the reply cannot be parsed — a truncated or prose reply is a
    failure, not an empty result. Handles models that reject `response_format` by retrying once
    without it and caching that fact per model.
    """
    client = get_client()
    target = model or get_active_model()

    msgs = [
        {"role": "system", "content": system_prompt + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation, just JSON."},
        {"role": "user", "content": user_prompt},
    ]

    def _create(use_json_mode: bool):
        kwargs = dict(model=target, messages=msgs, temperature=temperature, max_tokens=max_tokens)
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return _call_with_retry(lambda: client.chat.completions.create(**kwargs))

    use_json_mode = _supports_json_mode.get(target, True)
    try:
        response = _create(use_json_mode)
        if use_json_mode:
            _supports_json_mode[target] = True
    except Exception as exc:
        # Only an UNSUPPORTED-PARAMETER error justifies a second paid call. Previously any exception
        # here — including a usage-callback bug or a hard auth failure — silently triggered one.
        if not (use_json_mode and _is_json_mode_rejection(exc)):
            raise
        _supports_json_mode[target] = False
        response = _create(False)

    # Record usage BEFORE parsing, and never let a callback error look like an API failure.
    if on_usage and getattr(response, "usage", None):
        try:
            on_usage(response.usage)
        except Exception as exc:  # noqa: BLE001
            print(f"[llm] usage callback failed ({type(exc).__name__}: {exc})")

    choice = response.choices[0] if response.choices else None
    text = (choice.message.content if choice and choice.message else "") or ""
    parsed = _extract_json(text)
    if parsed is None:
        reason = ("reply was truncated (hit max_tokens)"
                  if choice and getattr(choice, "finish_reason", None) == "length"
                  else "reply was not JSON")
        raise JSONResponseError(f"{target}: {reason}", raw=text[:400])
    return parsed


def _is_json_mode_rejection(exc: Exception) -> bool:
    """True when the provider rejected `response_format`, rather than failing for another reason."""
    err = str(exc).lower()
    markers = ("response_format", "json_object", "json mode", "unsupported parameter",
               "unrecognized request argument", "not supported")
    return any(m in err for m in markers)
