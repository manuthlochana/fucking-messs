"""Thin async wrapper around the google-genai SDK for structured output.

Both the Sentry (classification) and the crawler (extraction) need to call
Gemini Flash and get back a validated Pydantic object, so the client lives in
one place. Verified against ``google-genai`` 2.10.0:

    client = genai.Client(api_key=...)
    resp = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=MyPydanticModel,          # Pydantic class accepted
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    obj = resp.parsed        # -> instance of MyPydanticModel
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from config import Settings, settings as default_settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the model cannot produce a valid structured response."""


class GeminiClient:
    """Async, retrying client that always returns a validated Pydantic model."""

    def __init__(self, settings: Settings = default_settings) -> None:
        self._settings = settings
        self._client = None  # lazily created so imports never need a key
        self._types = None

    def _ensure_client(self):
        if self._client is None:
            # Imported lazily: keeps `import llm` cheap and avoids hard-failing
            # in environments where the SDK isn't installed yet.
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=self._settings.require_api_key())
            self._types = types
        return self._client, self._types

    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        *,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        disable_thinking: Optional[bool] = None,
    ) -> T:
        """Call Gemini and return an instance of ``schema``.

        Retries transient failures with exponential backoff and, if the model
        rejects ``thinking_config`` (older/non-2.5 models), transparently
        retries once without it.
        """
        client, types = self._ensure_client()
        model_name = model or self._settings.gemini_model
        want_no_thinking = (
            self._settings.disable_thinking if disable_thinking is None else disable_thinking
        )

        def _build_config(include_thinking: bool):
            kwargs = dict(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=temperature,
            )
            if system_instruction:
                kwargs["system_instruction"] = system_instruction
            if max_output_tokens:
                kwargs["max_output_tokens"] = max_output_tokens
            if include_thinking and want_no_thinking:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            return types.GenerateContentConfig(**kwargs)

        include_thinking = True
        last_err: Optional[Exception] = None

        for attempt in range(self._settings.llm_max_retries + 1):
            try:
                resp = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=_build_config(include_thinking),
                )
                return self._coerce(resp, schema)
            except Exception as exc:  # noqa: BLE001 - normalize into LLMError below
                last_err = exc
                msg = str(exc).lower()
                # Some models don't accept thinking_config — drop it and retry now.
                if include_thinking and "think" in msg:
                    include_thinking = False
                    continue
                # Backoff for transient (rate limit / 5xx / network) errors.
                if attempt < self._settings.llm_max_retries:
                    await asyncio.sleep(0.75 * (2 ** attempt))
                    continue

        raise LLMError(f"structured generation failed after retries: {last_err}") from last_err

    def _coerce(self, resp, schema: Type[T]) -> T:
        """Prefer the SDK's parsed object; fall back to parsing raw text."""
        parsed = getattr(resp, "parsed", None)
        if isinstance(parsed, schema):
            return parsed
        if isinstance(parsed, dict):
            return schema.model_validate(parsed)
        text = getattr(resp, "text", None)
        if text:
            try:
                return schema.model_validate_json(text)
            except Exception:
                return schema.model_validate(json.loads(text))
        raise LLMError("model returned no parseable structured content")


# ===========================================================================
# Multi-key LLM pool with per-key rate limiting and automatic fallback
# ===========================================================================

import time as _time
from dataclasses import dataclass as _dataclass, field as _field
from typing import List as _List


@_dataclass
class _KeySlot:
    """Tracks one API key's state in the rotation pool."""

    provider: str          # "gemini" | "groq"
    api_key: str
    rpm_limit: int         # requests-per-minute ceiling from provider
    safety_margin: float   # fraction of rpm_limit to treat as ceiling
    # Runtime state (mutable).
    cooldown_until: float = _field(default=0.0)        # epoch seconds
    calls_this_window: int = _field(default=0)
    window_start: float = _field(default_factory=_time.monotonic)
    total_calls: int = _field(default=0)
    total_errors: int = _field(default=0)

    @property
    def effective_rpm(self) -> int:
        return max(1, int(self.rpm_limit * self.safety_margin))

    def is_available(self) -> bool:
        """True when the key is not on cooldown and under the RPM safety ceiling."""
        now = _time.monotonic()
        if now < self.cooldown_until:
            return False
        elapsed = now - self.window_start
        if elapsed >= 60.0:
            # New 60-second window.
            self.calls_this_window = 0
            self.window_start = now
        return self.calls_this_window < self.effective_rpm

    def record_call(self) -> None:
        self.calls_this_window += 1
        self.total_calls += 1

    def mark_rate_limited(self, backoff_s: float = 65.0) -> None:
        """Enter hard cooldown after a 429 / rate-limit response."""
        self.cooldown_until = _time.monotonic() + backoff_s
        self.total_errors += 1

    def mark_error(self) -> None:
        self.total_errors += 1


class MultiKeyLLMPool:
    """Resilient multi-provider LLM pool with automatic key rotation.

    Load order:
    1. All keys from ``GEMINI_API_KEYS`` (comma-separated) as Gemini slots.
    2. Single ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY`` as final Gemini fallback.
    3. ``GROQ_API_KEY`` as the lowest-priority fallback slot.

    On each ``generate_structured`` call the pool:
    - Picks the first available slot (not on cooldown, under RPM ceiling).
    - On 429 / timeout: marks the slot on cooldown, advances to the next.
    - All slots exhausted → raises ``LLMError``.
    - Validates every response against the Pydantic schema before returning.

    Usage::

        pool = MultiKeyLLMPool.from_settings(settings)
        result = await pool.generate_structured(prompt, MySchema)
    """

    def __init__(self, slots: "_List[_KeySlot]") -> None:
        if not slots:
            raise ValueError("MultiKeyLLMPool requires at least one API key slot.")
        self._slots = slots
        self._gemini_clients: dict[str, object] = {}
        self._groq_clients: dict[str, object] = {}
        self._types = None

    @classmethod
    def from_settings(cls, settings: "Settings") -> "MultiKeyLLMPool":  # noqa: F821
        """Build a pool from application settings."""
        slots: _List[_KeySlot] = []
        seen_keys: set[str] = set()

        # Primary Gemini key(s) from GEMINI_API_KEYS.
        for key in settings.gemini_api_keys:
            if key not in seen_keys:
                slots.append(
                    _KeySlot(
                        provider="gemini",
                        api_key=key,
                        rpm_limit=settings.gemini_rpm_limit,
                        safety_margin=settings.llm_rpm_safety_margin,
                    )
                )
                seen_keys.add(key)

        # Groq fallback.
        if settings.groq_api_key and settings.groq_api_key not in seen_keys:
            slots.append(
                _KeySlot(
                    provider="groq",
                    api_key=settings.groq_api_key,
                    rpm_limit=settings.groq_rpm_limit,
                    safety_margin=settings.llm_rpm_safety_margin,
                )
            )
            seen_keys.add(settings.groq_api_key)

        if not slots:
            # Last-ditch: use the single-key field so existing code keeps working.
            key = settings.gemini_api_key
            if key:
                slots.append(
                    _KeySlot(
                        provider="gemini",
                        api_key=key,
                        rpm_limit=settings.gemini_rpm_limit,
                        safety_margin=settings.llm_rpm_safety_margin,
                    )
                )
        return cls(slots)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def generate_structured(
        self,
        prompt: str,
        schema: "Type[T]",  # noqa: F821
        *,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        disable_thinking: Optional[bool] = None,
        model: Optional[str] = None,
    ) -> "T":  # noqa: F821
        """Call the best available LLM slot and return a validated Pydantic object."""
        last_err: Optional[Exception] = None

        for slot in self._slots:
            if not slot.is_available():
                continue
            slot.record_call()
            try:
                if slot.provider == "gemini":
                    return await self._call_gemini(
                        slot, prompt, schema,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        disable_thinking=disable_thinking,
                        model=model,
                    )
                elif slot.provider == "groq":
                    return await self._call_groq(
                        slot, prompt, schema,
                        system_instruction=system_instruction,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
            except LLMError:
                raise  # Schema validation failures propagate immediately.
            except Exception as exc:
                last_err = exc
                msg = str(exc).lower()
                if "429" in msg or "rate" in msg or "quota" in msg or "exhausted" in msg:
                    slot.mark_rate_limited()
                else:
                    slot.mark_error()
                continue

        raise LLMError(
            f"All LLM pool slots exhausted or on cooldown. Last error: {last_err}"
        ) from last_err

    def slot_summary(self) -> str:
        """Return a human-readable status of all slots (for logging)."""
        lines = []
        for i, s in enumerate(self._slots):
            avail = "ok" if s.is_available() else "cooldown"
            lines.append(
                f"  slot[{i}] {s.provider} ...{s.api_key[-6:]} "
                f"calls={s.total_calls} errs={s.total_errors} [{avail}]"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Provider-specific call helpers
    # ------------------------------------------------------------------ #

    def _gemini_client(self, key: str):
        if key not in self._gemini_clients:
            from google import genai
            from google.genai import types
            self._gemini_clients[key] = genai.Client(api_key=key)
            self._types = types
        return self._gemini_clients[key], self._types

    async def _call_gemini(
        self, slot: _KeySlot, prompt: str, schema,
        *, system_instruction, temperature, max_output_tokens, disable_thinking, model
    ):
        from config import settings as _settings
        client, types = self._gemini_client(slot.api_key)
        model_name = model or _settings.gemini_model
        want_no_thinking = _settings.disable_thinking if disable_thinking is None else disable_thinking

        cfg_kwargs: dict = dict(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        )
        if system_instruction:
            cfg_kwargs["system_instruction"] = system_instruction
        if max_output_tokens:
            cfg_kwargs["max_output_tokens"] = max_output_tokens
        if want_no_thinking:
            try:
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass

        cfg = types.GenerateContentConfig(**cfg_kwargs)
        resp = await client.aio.models.generate_content(
            model=model_name, contents=prompt, config=cfg
        )
        # Reuse GeminiClient._coerce logic.
        _gc = GeminiClient.__new__(GeminiClient)
        return _gc._coerce(resp, schema)

    def _groq_client(self, key: str):
        if key not in self._groq_clients:
            try:
                from groq import AsyncGroq
                self._groq_clients[key] = AsyncGroq(api_key=key)
            except ImportError:
                raise LLMError("groq package not installed. Run: pip install groq")
        return self._groq_clients[key]

    async def _call_groq(
        self, slot: _KeySlot, prompt: str, schema,
        *, system_instruction, temperature, max_output_tokens
    ):
        import json
        from config import settings as _settings
        client = self._groq_client(slot.api_key)

        # Build a JSON-mode prompt embedding the Pydantic JSON schema.
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_msg = (
            (system_instruction or "You are a helpful assistant.") +
            f"\n\nYou MUST respond with a valid JSON object matching this schema:\n{schema_json}"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        resp = await client.chat.completions.create(
            model=_settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens or 1024,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        try:
            return schema.model_validate_json(content)
        except Exception:
            return schema.model_validate(json.loads(content))
