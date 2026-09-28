"""Compatibility alias for :mod:`forensic_worker`.

The Track B forensic pipeline is implemented in :mod:`forensic_worker`; some
blueprint references and call-sites spell the module ``forensics``. This shim
re-exports the public pipeline API so both import spellings resolve to the same
objects.

Prefer importing from :mod:`forensic_worker` directly in new code.
"""

from __future__ import annotations

from forensic_worker import (  # noqa: F401  (re-exported for API compatibility)
    TOTAL_FORENSIC_BUDGET_S,
    ForensicContext,
    ForensicReport,
    run_forensic_pipeline,
    run_worker_loop,
)

__all__ = [
    "TOTAL_FORENSIC_BUDGET_S",
    "ForensicContext",
    "ForensicReport",
    "run_forensic_pipeline",
    "run_worker_loop",
]
