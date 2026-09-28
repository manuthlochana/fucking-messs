"""Compatibility alias for :mod:`normalizer`.

Some blueprint sections and older call-sites refer to the spec-canonicalisation
layer as ``canonical`` (the thing that turns a messy scraped title into a
canonical product fingerprint). The implementation lives in :mod:`normalizer`;
this module simply re-exports its public API so ``import canonical`` and
``from canonical import normalizer`` both keep working.

Prefer importing from :mod:`normalizer` directly in new code.
"""

from __future__ import annotations

from normalizer import (  # noqa: F401  (re-exported for API compatibility)
    NormalizedSpec,
    SpecNormalizer,
    UniversalTechSpec,
    UniversalTechSpecExtractor,
    normalizer,
)

#: The shared singleton, aliased under the historical ``canonical`` name.
canonical = normalizer

__all__ = [
    "NormalizedSpec",
    "SpecNormalizer",
    "UniversalTechSpec",
    "UniversalTechSpecExtractor",
    "normalizer",
    "canonical",
]
