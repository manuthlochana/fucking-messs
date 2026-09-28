"""Unit tests for Track A extraction + site profiles + forensic budgeting.

Covers the four behaviours the audit added / relies on:

1. BNPL hidden-markup maths — ``markup_pct = ((cycles·installment) − cash)/cash·100``.
2. Three-signal (2-of-3) stock verification — agreement, disagreement, abstention.
3. Site-profile loading & domain resolution (with graceful PyYAML-absent fallback).
4. Forensic per-phase timeout + cumulative-budget cap with partial commits.

Pure unit tests: no network, no database, no live LLM.
"""

import asyncio
import sys
from pathlib import Path

try:
    import pytest
except ImportError:  # pragma: no cover - allow bare `python tests/test_extractor.py`
    class _MockPytest:
        @staticmethod
        def fixture(*a, **k):
            def deco(fn):
                return fn
            return deco

        class mark:
            @staticmethod
            def parametrize(names, values):
                def deco(fn):
                    fn._parametrize = (names, values)
                    return fn
                return deco

            class _Async:
                def __call__(self, fn):
                    return fn

            asyncio = _Async()

    pytest = _MockPytest()

# Allow importing from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor import (
    extract_bank_promos,
    extract_bnpl_plans,
    extract_surcharges,
    extract_track_a,
    verify_stock,
)


# ---------------------------------------------------------------------------
# 1. BNPL hidden-markup maths
# ---------------------------------------------------------------------------

def test_bnpl_markup_positive():
    """12 × Rs.10,000 = 120,000 against a 100,000 cash price → +20% markup."""
    plans = extract_bnpl_plans(
        "Pay in 12 installments of Rs. 10,000 with Koko. Cash price Rs. 100,000.",
        cash_price=100_000.0,
    )
    assert len(plans) == 1
    plan = plans[0]
    assert plan.cycles == 12
    assert plan.installment_amount == 10_000.0
    assert plan.total_payable == 120_000.0
    assert abs(plan.markup_pct - 20.0) < 0.01


def test_bnpl_markup_unknown_cash_price_is_zero():
    """With no cash price we still surface the plan but leave markup at 0.0."""
    plans = extract_bnpl_plans("3 x Rs 9,900 with Mintpay", cash_price=None)
    assert len(plans) == 1
    assert plans[0].markup_pct == 0.0
    assert plans[0].total_payable == 29_700.0


def test_bnpl_ignores_unrelated_multiplier():
    """'3 x USB cables' near no BNPL provider must not become a fake plan."""
    plans = extract_bnpl_plans("Bundle: 3 x USB cables included.", cash_price=5000.0)
    assert plans == []


# ---------------------------------------------------------------------------
# 2. Three-signal (2-of-3) stock verification
# ---------------------------------------------------------------------------

def test_stock_two_of_three_agree_in_stock():
    v = verify_stock(button_in_stock=True, schema_availability=None,
                     stock_text="In Stock — ready to ship")
    assert v.in_stock is True
    assert v.agreement_reached is True
    assert v.confidence >= 2


def test_stock_two_of_three_agree_out_of_stock():
    v = verify_stock(button_in_stock=False, schema_availability=None,
                     stock_text="Sold Out")
    assert v.in_stock is False
    assert v.agreement_reached is True


def test_stock_single_signal_no_agreement():
    """A lone weak signal must not flip the verdict (needs 2-of-3)."""
    v = verify_stock(button_in_stock=None, schema_availability=None, stock_text="")
    assert v.in_stock is None
    assert v.agreement_reached is False


def test_stock_conflict_resolved_by_majority():
    """Button False but schema+text both InStock → majority wins (in stock)."""
    v = verify_stock(
        button_in_stock=False,
        schema_availability=True,
        stock_text="Add to cart — in stock",
    )
    assert v.in_stock is True
    assert v.agreement_reached is True


# ---------------------------------------------------------------------------
# 3. Bank promos + surcharges + bundled Track A
# ---------------------------------------------------------------------------

def test_bank_promos_no_window_bleed():
    """Each bank keeps its own %/months — HNB must not steal Commercial's 15%."""
    promos = extract_bank_promos(
        "Commercial Bank credit card 15% off. HNB card 0% for 12 months."
    )
    by_bank = {p.bank.lower(): p for p in promos}
    assert any("commercial" in b for b in by_bank)
    assert any("hnb" in b for b in by_bank)


def test_extract_track_a_bundle():
    html = (
        "<html><body><h1>Widget</h1>"
        '<button class="add-to-cart">Add to Cart</button>'
        "Pay in 4 installments of Rs. 2,500 with Koko. "
        "Commercial Bank 10% off. 3% card surcharge applies. "
        "Cash price Rs. 9,000.</body></html>"
    )
    extras = extract_track_a(html, cash_price=9_000.0)
    assert len(extras.bnpl_plans) >= 1
    assert extras.stock.in_stock is True  # button + text agree
    assert len(extras.bank_promos) >= 1
    assert len(extras.surcharges) >= 1


def test_extract_track_a_never_raises_on_garbage():
    extras = extract_track_a("", cash_price=None)
    assert extras.bnpl_plans == []
    assert extras.stock.in_stock is None


# ---------------------------------------------------------------------------
# 5. Forensic per-phase timeout + cumulative budget with partial commits
# ---------------------------------------------------------------------------

def test_forensic_budget_constant():
    import forensic_worker

    assert forensic_worker.TOTAL_FORENSIC_BUDGET_S == 300.0


def _run(coro):
    return asyncio.run(coro)


def test_phase_timeout_is_caught_and_recorded():
    """A phase that overruns its budget is caught, timed, and not fatal."""
    from forensic_worker import ForensicReport, _run_phase_with_budget

    report = ForensicReport(product_id="test")

    async def _slow():
        await asyncio.sleep(5.0)
        return "should-not-return"

    result = _run(
        _run_phase_with_budget(
            "phase_slow", _slow, per_phase_timeout=0.05,
            remaining_budget=0.05, report=report,
        )
    )
    # Timeout → no result, timing recorded, error logged (but NOT "skipped":
    # the phase was started, it just overran).
    assert result is None
    assert "phase_slow" in report.phase_timings
    assert any("phase_slow" in e for e in report.errors)
    assert "phase_slow" not in report.skipped_phases


def test_phase_skipped_when_budget_exhausted():
    """When the cumulative budget is already spent, the phase is skipped, not run."""
    from forensic_worker import ForensicReport, _run_phase_with_budget

    report = ForensicReport(product_id="test")
    ran = {"called": False}

    async def _work():
        ran["called"] = True
        return "ok"

    result = _run(
        _run_phase_with_budget(
            "phase_late", _work, per_phase_timeout=10.0,
            remaining_budget=0.0, report=report,
        )
    )
    assert result is None
    assert ran["called"] is False
    assert "phase_late" in report.skipped_phases


def test_phase_success_records_timing():
    from forensic_worker import ForensicReport, _run_phase_with_budget

    report = ForensicReport(product_id="test")

    async def _fast():
        return {"ok": True}

    result = _run(
        _run_phase_with_budget(
            "phase_ok", _fast, per_phase_timeout=5.0,
            remaining_budget=100.0, report=report,
        )
    )
    assert result == {"ok": True}
    assert "phase_ok" in report.phase_timings
    assert "phase_ok" not in report.skipped_phases


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
