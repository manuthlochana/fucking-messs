"""Site profile package: per-domain scraping selector configs.

A *site profile* is a small declarative bundle of CSS/XPath selectors (plus a
few flags) that tells the ingestion layer where to find the title, price, stock
button, spec table, card-promo banners and pagination links on a given
retailer's product page. Profiles are plain YAML files in this directory and are
loaded by :mod:`site_profiles.loader`.

The loader degrades gracefully when PyYAML is not installed (it falls back to a
tiny built-in generic profile), so importing this package never hard-fails.
"""

from __future__ import annotations

from site_profiles.loader import (  # noqa: F401
    SiteProfile,
    get_profile,
    list_profiles,
    load_all_profiles,
    load_profile_file,
)

__all__ = [
    "SiteProfile",
    "get_profile",
    "list_profiles",
    "load_all_profiles",
    "load_profile_file",
]
