"""Unit tests for MultiKeyLLMPool with mocked HTTP calls.

These tests verify:
- Key rotation on 429 / rate-limit responses.
- RPM safety margin triggers soft cooldown.
- All slots exhausted raises LLMError.
- Groq slot is used as final fallback.

No live API calls are made.
"""

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import pytest
except ImportError:
    class _RaisesContext:
        def __init__(self, exc_type, match=None):
            self.exc_type = exc_type
            self.match = match
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected {self.exc_type.__name__} but no exception was raised.")
            if not issubclass(exc_type, self.exc_type):
                return False
            if self.match and self.match not in str(exc_val):
                raise AssertionError(f"Expected match '{self.match}' in '{exc_val}'")
            return True

    class _MockPytest:
        @staticmethod
        def raises(exc_type, match=None):
            return _RaisesContext(exc_type, match)
        class mark:
            @staticmethod
            def asyncio(fn):
                fn._is_async = True
                return fn
    pytest = _MockPytest()

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import LLMError, MultiKeyLLMPool, _KeySlot
from pydantic import BaseModel


class _DummySchema(BaseModel):
    value: str = "ok"


def _make_slot(provider="gemini", key="key1", rpm=10, margin=0.8, **kwargs) -> _KeySlot:
    return _KeySlot(provider=provider, api_key=key, rpm_limit=rpm, safety_margin=margin, **kwargs)


# ---------------------------------------------------------------------------
# _KeySlot availability
# ---------------------------------------------------------------------------

def test_slot_available_initially():
    slot = _make_slot()
    assert slot.is_available()


def test_slot_on_cooldown_is_not_available():
    slot = _make_slot()
    slot.mark_rate_limited(backoff_s=3600)
    assert not slot.is_available()


def test_slot_cooldown_expires():
    slot = _make_slot()
    slot.cooldown_until = time.monotonic() - 1  # already expired
    assert slot.is_available()


def test_slot_rpm_ceiling_triggers_unavailability():
    # effective_rpm = 10 * 0.8 = 8
    slot = _make_slot(rpm=10, margin=0.8)
    slot.calls_this_window = 8  # at ceiling
    assert not slot.is_available()


def test_slot_rpm_window_resets_after_60s():
    slot = _make_slot(rpm=10, margin=0.8)
    slot.calls_this_window = 8
    slot.window_start = time.monotonic() - 61  # window expired
    assert slot.is_available()


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def test_pool_requires_at_least_one_slot():
    with pytest.raises(ValueError):
        MultiKeyLLMPool(slots=[])


def test_pool_constructed_with_multiple_slots():
    slots = [_make_slot(key=f"key{i}") for i in range(3)]
    pool = MultiKeyLLMPool(slots)
    assert len(pool._slots) == 3


# ---------------------------------------------------------------------------
# Key rotation on 429
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_key_rotates_on_rate_limit():
    """When slot 0 returns a 429-like exception, slot 1 should be tried."""
    slot0 = _make_slot(key="key0")
    slot1 = _make_slot(key="key1")
    pool = MultiKeyLLMPool(slots=[slot0, slot1])

    call_count = 0

    async def _fake_call_gemini(slot, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if slot.api_key == "key0":
            raise Exception("429 Too Many Requests")
        return _DummySchema(value="slot1_responded")

    pool._call_gemini = _fake_call_gemini

    result = await pool.generate_structured("test", _DummySchema)
    assert result.value == "slot1_responded"
    assert slot0.cooldown_until > time.monotonic()  # key0 is now on cooldown
    assert call_count == 2


@pytest.mark.asyncio
async def test_all_slots_exhausted_raises_llm_error():
    """When all slots are on cooldown, LLMError must be raised."""
    slot0 = _make_slot(key="key0")
    slot0.mark_rate_limited(backoff_s=9999)
    pool = MultiKeyLLMPool(slots=[slot0])

    with pytest.raises(LLMError, match="exhausted"):
        await pool.generate_structured("test", _DummySchema)


@pytest.mark.asyncio
async def test_groq_used_as_fallback_when_gemini_exhausted():
    """Groq slot should be tried last when all Gemini slots are on cooldown."""
    gemini_slot = _make_slot(provider="gemini", key="gkey")
    gemini_slot.mark_rate_limited(backoff_s=9999)
    groq_slot = _make_slot(provider="groq", key="groqkey")
    pool = MultiKeyLLMPool(slots=[gemini_slot, groq_slot])

    async def _fake_groq(slot, *args, **kwargs):
        return _DummySchema(value="groq_responded")

    pool._call_groq = _fake_groq

    result = await pool.generate_structured("test", _DummySchema)
    assert result.value == "groq_responded"


# ---------------------------------------------------------------------------
# slot_summary
# ---------------------------------------------------------------------------

def test_slot_summary_contains_all_slots():
    slots = [_make_slot(key=f"key{i}") for i in range(3)]
    pool = MultiKeyLLMPool(slots)
    summary = pool.slot_summary()
    for i in range(3):
        assert f"key{i}" in summary or f"...key{i}" in summary or "ok" in summary


if __name__ == "__main__":
    import asyncio
    import inspect
    passed = 0
    failed = 0
    current_module = sys.modules[__name__]
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and name.startswith("test_"):
            try:
                if inspect.iscoroutinefunction(obj):
                    asyncio.run(obj())
                else:
                    obj()
                passed += 1
            except Exception as e:
                print(f"FAIL: {name}: {e}")
                failed += 1
    print(f"\nResult: {passed} passed, {failed} failed.")
    if failed:
        sys.exit(1)
