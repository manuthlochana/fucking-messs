#!/usr/bin/env python3
"""KALA-BALANA preflight verifier.

A dependency-light, offline sanity gate you can run before deploying or after a
refactor. It never touches the network or a live database. Checks, in order:

  1. Byte-compile every first-party module (syntax gate).
  2. Import the core modules (import-time gate).
  3. Confirm schema.sql declares the expected tables.
  4. Load the YAML site profiles (with graceful PyYAML-absent degradation).
  5. Smoke-test the deterministic Track A extractor (BNPL markup + 2-of-3 stock).

Exit code is 0 when every check passes, 1 otherwise — suitable for CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (label, ok, detail) tuples accumulated as we go.
_results: list[tuple[str, bool, str]] = []


def _record(label: str, ok: bool, detail: str = "") -> None:
    _results.append((label, ok, detail))
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def check_compile() -> None:
    import py_compile

    print("\n[1/5] Byte-compiling first-party modules…")
    modules = sorted(p for p in ROOT.glob("*.py") if p.name != "verify_preflight.py")
    modules += sorted((ROOT / "db").glob("*.py"))
    modules += sorted((ROOT / "site_profiles").glob("*.py"))
    failures = 0
    for m in modules:
        try:
            py_compile.compile(str(m), doraise=True)
        except py_compile.PyCompileError as exc:
            failures += 1
            _record(f"compile {m.relative_to(ROOT)}", False, str(exc).splitlines()[-1])
    if failures == 0:
        _record(f"compile {len(modules)} module(s)", True)


def check_imports() -> None:
    print("\n[2/5] Importing core modules…")
    core = [
        "config",
        "schemas",
        "normalizer",
        "canonical",
        "extractor",
        "forensic_worker",
        "forensics",
        "advisor",
        "site_profiles.loader",
    ]
    for name in core:
        try:
            __import__(name)
            _record(f"import {name}", True)
        except Exception as exc:  # pragma: no cover - reported, not raised
            _record(f"import {name}", False, f"{type(exc).__name__}: {exc}")


def check_schema() -> None:
    print("\n[3/5] Inspecting schema.sql…")
    schema = ROOT / "schema.sql"
    if not schema.exists():
        _record("schema.sql present", False, "file missing")
        return
    sql = schema.read_text(encoding="utf-8").lower()
    expected = [
        "canonical_products",
        "product_embeddings",
        "merchants",
        "merchant_forensics",
        "listings",
        "defect_dossiers",
        "component_teardowns",
        "forensic_queue",
        "price_history",
        "currency_arbitrage_logs",
    ]
    missing = [t for t in expected if f"table if not exists {t}" not in sql]
    _record("schema tables declared", not missing,
            "missing: " + ", ".join(missing) if missing else f"{len(expected)} tables")
    _record("hnsw embedding index", "using hnsw" in sql)
    _record("partition maintenance fn", "ensure_monthly_partitions" in sql)


def check_profiles() -> None:
    print("\n[4/5] Loading site profiles…")
    try:
        from site_profiles import loader

        profiles = loader.load_all_profiles(force=True)
        names = loader.list_profiles()
        _record("site profiles loaded", len(profiles) >= 1,
                f"{len(profiles)} profile(s): {', '.join(sorted(names))}")
        # Domain resolution must always return *something* (generic fallback).
        prof = loader.get_profile("https://sampleretailer.lk/product/x")
        _record("get_profile resolves", prof is not None,
                getattr(prof, "name", "?"))
        if not getattr(loader, "_HAVE_YAML", True):
            _record("PyYAML-absent degradation", True, "builtin generic fallback active")
    except Exception as exc:
        _record("site profiles", False, f"{type(exc).__name__}: {exc}")


def check_extractor() -> None:
    print("\n[5/5] Smoke-testing Track A extractor…")
    try:
        from extractor import extract_bnpl_plans, verify_stock

        # BNPL hidden-markup: 12 × Rs.10,000 = 120,000 vs 100,000 cash = +20%.
        plans = extract_bnpl_plans(
            "Pay in 12 installments of Rs. 10,000 with Koko. Cash price Rs. 100,000.",
            cash_price=100_000.0,
        )
        got_markup = any(abs((p.markup_pct or 0) - 20.0) < 0.5 for p in plans)
        _record("BNPL markup ≈ 20%", got_markup,
                f"{len(plans)} plan(s)")

        # 2-of-3 agreement: button=True + stock text 'in stock' → verdict True.
        v_true = verify_stock(button_in_stock=True, schema_availability=None,
                              stock_text="In Stock — ready to ship")
        _record("2-of-3 stock → in stock", v_true.in_stock is True,
                f"confidence={v_true.confidence}")

        # Single weak signal must NOT reach agreement.
        v_none = verify_stock(button_in_stock=None, schema_availability=None,
                              stock_text="")
        _record("no signal → no verdict", v_none.in_stock is None
                and not v_none.agreement_reached)
    except Exception as exc:
        _record("extractor smoke test", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    print("=" * 68)
    print(" KALA-BALANA PREFLIGHT VERIFIER")
    print("=" * 68)

    check_compile()
    check_imports()
    check_schema()
    check_profiles()
    check_extractor()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print("\n" + "=" * 68)
    print(f" RESULT: {passed} passed, {failed} failed")
    print("=" * 68)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
