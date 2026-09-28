"""YAML site-profile loader with graceful degradation.

A :class:`SiteProfile` maps a retailer domain to the selectors ingestion needs.
Profiles live as ``*.yaml`` files next to this module. Loading is:

* **Lazy & cached** — profiles are parsed once and memoised.
* **Dependency-light** — if PyYAML is unavailable the loader falls back to a
  minimal built-in generic profile so callers never crash on a missing dep.
* **Domain-matched** — :func:`get_profile` resolves the best profile for a URL
  by longest matching ``domains`` suffix, falling back to ``default_generic``.

The extractor/crawler treat every selector field as *optional*: a profile that
omits a selector simply means "fall back to the generic heuristics".
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from logging_utils import log

_PROFILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Optional dependency — never a hard requirement.
try:  # pragma: no cover - trivial
    import yaml  # type: ignore

    _HAVE_YAML = True
except Exception:  # pragma: no cover - yaml simply not installed
    yaml = None  # type: ignore
    _HAVE_YAML = False


@dataclass
class SiteProfile:
    """Declarative per-domain selector bundle.

    Every selector is optional. ``selector_type`` records whether the selector
    strings are CSS (default) or XPath, so the consumer knows how to apply them.
    """

    name: str
    domains: List[str] = field(default_factory=list)
    selector_type: str = "css"  # "css" | "xpath"

    title: Optional[str] = None
    price: Optional[str] = None
    original_price: Optional[str] = None
    stock_button: Optional[str] = None
    stock_text: Optional[str] = None
    specs_table: Optional[str] = None
    card_promo: Optional[str] = None
    pagination: Optional[str] = None

    # Free-form extras (JSON-LD hints, wait selectors, currency, notes…).
    extra: Dict[str, object] = field(default_factory=dict)

    def matches(self, host: str) -> Optional[int]:
        """Return the length of the longest matching domain suffix, or None.

        Longer matches are more specific and win, so ``store.example.com`` beats
        a bare ``example.com`` entry.
        """
        host = (host or "").lower().lstrip(".")
        best: Optional[int] = None
        for dom in self.domains:
            dom = str(dom).lower().lstrip(".")
            if host == dom or host.endswith("." + dom):
                if best is None or len(dom) > best:
                    best = len(dom)
        return best

    @classmethod
    def from_dict(cls, data: Dict[str, object], *, name_hint: str = "") -> "SiteProfile":
        """Build a profile from a parsed YAML mapping, tolerating unknown keys."""
        data = dict(data or {})
        selectors = dict(data.get("selectors") or {})  # type: ignore[arg-type]
        known = {
            "title", "price", "original_price", "stock_button",
            "stock_text", "specs_table", "card_promo", "pagination",
        }
        # Selectors may be nested under `selectors:` or given flat at top level.
        resolved = {k: (selectors.get(k) or data.get(k)) for k in known}
        domains = data.get("domains") or data.get("domain") or []
        if isinstance(domains, str):
            domains = [domains]
        extra = {
            k: v for k, v in data.items()
            if k not in known | {"name", "domains", "domain", "selectors", "selector_type"}
        }
        return cls(
            name=str(data.get("name") or name_hint or "unnamed"),
            domains=[str(d) for d in domains],
            selector_type=str(data.get("selector_type") or "css"),
            title=resolved["title"],
            price=resolved["price"],
            original_price=resolved["original_price"],
            stock_button=resolved["stock_button"],
            stock_text=resolved["stock_text"],
            specs_table=resolved["specs_table"],
            card_promo=resolved["card_promo"],
            pagination=resolved["pagination"],
            extra=extra,
        )


#: Built-in fallback used when PyYAML is missing or no file matches.
_BUILTIN_GENERIC = SiteProfile(
    name="default_generic",
    domains=[],
    selector_type="css",
    title="h1, [itemprop='name'], .product-title",
    price="[itemprop='price'], .price, .product-price, .amount",
    original_price="del .amount, .regular-price, .was-price, s .amount",
    stock_button="button.add-to-cart, button[name='add'], .single_add_to_cart_button",
    stock_text=".stock, .availability, [itemprop='availability']",
    specs_table="table.specs, table.shop_attributes, .woocommerce-product-attributes",
    card_promo=".promo, .bank-offer, .card-offer, .payment-banner",
    pagination="a.next, .pagination a, .page-numbers a",
)

# Module-level memo caches.
_CACHE: Optional[List[SiteProfile]] = None
_BY_NAME: Dict[str, SiteProfile] = {}


def load_profile_file(path: str) -> Optional[SiteProfile]:
    """Parse a single YAML profile file into a :class:`SiteProfile`.

    Returns ``None`` (with a logged warning) when YAML is unavailable or the
    file is malformed — a bad profile never takes down the loader.
    """
    if not _HAVE_YAML:
        log.debug(f"PyYAML missing; cannot parse profile {os.path.basename(path)}.")
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            log.warn(f"Profile {os.path.basename(path)} is not a mapping; skipped.")
            return None
        name_hint = os.path.splitext(os.path.basename(path))[0]
        return SiteProfile.from_dict(data, name_hint=name_hint)
    except Exception as exc:  # pragma: no cover - defensive
        log.warn(f"Failed to load site profile {os.path.basename(path)}: {exc!r}")
        return None


def load_all_profiles(force: bool = False) -> List[SiteProfile]:
    """Load and memoise every ``*.yaml`` profile in the package directory.

    Always includes a generic fallback profile so :func:`get_profile` can never
    return ``None``.
    """
    global _CACHE, _BY_NAME
    if _CACHE is not None and not force:
        return _CACHE

    profiles: List[SiteProfile] = []
    if _HAVE_YAML and os.path.isdir(_PROFILE_DIR):
        for fname in sorted(os.listdir(_PROFILE_DIR)):
            if not fname.endswith((".yaml", ".yml")):
                continue
            prof = load_profile_file(os.path.join(_PROFILE_DIR, fname))
            if prof is not None:
                profiles.append(prof)

    # Guarantee a generic fallback is present exactly once.
    if not any(p.name == "default_generic" for p in profiles):
        profiles.append(_BUILTIN_GENERIC)

    _CACHE = profiles
    _BY_NAME = {p.name: p for p in profiles}
    log.debug(f"Loaded {len(profiles)} site profile(s): {list(_BY_NAME)}")
    return profiles


def list_profiles() -> List[str]:
    """Return the names of all loaded profiles."""
    load_all_profiles()
    return list(_BY_NAME.keys())


def get_profile(url_or_host: str) -> SiteProfile:
    """Resolve the best profile for a URL or bare host.

    Picks the profile whose ``domains`` has the longest matching suffix; falls
    back to the generic profile when nothing matches.
    """
    profiles = load_all_profiles()
    host = url_or_host or ""
    if "//" in host or "/" in host:
        host = urlsplit(host if "//" in host else "//" + host).netloc or host
    host = host.split(":")[0].lower()

    best: Optional[SiteProfile] = None
    best_score = -1
    for prof in profiles:
        score = prof.matches(host)
        if score is not None and score > best_score:
            best, best_score = prof, score

    if best is not None:
        return best
    return _BY_NAME.get("default_generic", _BUILTIN_GENERIC)
