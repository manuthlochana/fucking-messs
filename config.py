"""Central configuration for the Smart Sentry scraping engine.

All tunables are sourced from environment variables (optionally via a local
``.env`` file) so the same code runs unchanged on a laptop or a 4 GB VPS.

Nothing here performs network I/O or constructs heavy objects at import time,
which keeps ``schemas``/``sentry`` importable in unit tests without secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Optional .env loading. python-dotenv is a convenience, never a hard dep.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv simply not installed
    pass


def _env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(key)
    return val if val not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# Chromium flags that shrink the per-tab memory footprint. Images are disabled
# both here (Blink setting) and again at the network layer via a route hook in
# ``crawler.py`` — belt and suspenders for the "4 GB VPS safe" target.
_DEFAULT_CHROME_ARGS: List[str] = [
    "--disable-gpu",
    "--disable-dev-shm-usage",  # avoid /dev/shm exhaustion on small VPS
    "--disable-software-rasterizer",
    "--disable-extensions",
    "--blink-settings=imagesEnabled=false",
    "--mute-audio",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-background-timer-throttling",
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-breakpad",
    "--disable-sync",
    "--metrics-recording-only",
    "--disable-features=IsolateOrigins,site-per-process,TranslateUI,"
    "AudioServiceOutOfProcess,MediaRouter",
    "--js-flags=--max-old-space-size=512",  # cap V8 heap per renderer
]


@dataclass
class Settings:
    """Immutable-ish runtime configuration snapshot."""

    # --- Gemini / google-genai -------------------------------------------- #
    # google-genai reads GEMINI_API_KEY or GOOGLE_API_KEY automatically, but we
    # resolve it explicitly so we can fail fast with a helpful message.
    gemini_api_key: Optional[str] = field(
        default_factory=lambda: _env_str("GEMINI_API_KEY") or _env_str("GOOGLE_API_KEY")
    )
    # Fast, cheap classifier + extractor. Override with e.g. gemini-2.0-flash.
    gemini_model: str = field(default_factory=lambda: _env_str("GEMINI_MODEL", "gemini-2.5-flash"))
    # Disable "thinking" on the Sentry hot path for latency (2.5 Flash feature).
    disable_thinking: bool = field(default_factory=lambda: _env_bool("GEMINI_DISABLE_THINKING", True))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2))

    # --- Sentry thresholds ------------------------------------------------- #
    # The LLM only ever sees the first N tokens of cleaned content.
    sentry_token_budget: int = field(default_factory=lambda: _env_int("SENTRY_TOKEN_BUDGET", 2000))
    chars_per_token: int = field(default_factory=lambda: _env_int("CHARS_PER_TOKEN", 4))
    # Below this raw-DOM length the page is treated as empty/junk pre-LLM.
    min_dom_length: int = field(default_factory=lambda: _env_int("MIN_DOM_LENGTH", 500))

    # --- Crawler guardrails (anti-fragility) ------------------------------ #
    max_concurrency: int = field(default_factory=lambda: _env_int("MAX_CONCURRENCY", 2))
    max_pages: int = field(default_factory=lambda: _env_int("MAX_PAGES", 50))
    max_depth: int = field(default_factory=lambda: _env_int("MAX_DEPTH", 2))
    max_children_per_category: int = field(
        default_factory=lambda: _env_int("MAX_CHILDREN_PER_CATEGORY", 20)
    )

    # --- Browser / network ------------------------------------------------- #
    headless: bool = field(default_factory=lambda: _env_bool("HEADLESS", True))
    page_timeout_ms: int = field(default_factory=lambda: _env_int("PAGE_TIMEOUT_MS", 30_000))
    stealth_page_timeout_ms: int = field(
        default_factory=lambda: _env_int("STEALTH_PAGE_TIMEOUT_MS", 45_000)
    )
    stealth_settle_ms: int = field(default_factory=lambda: _env_int("STEALTH_SETTLE_MS", 2_500))
    proxy: Optional[str] = field(default_factory=lambda: _env_str("PROXY"))
    user_agent: Optional[str] = field(default_factory=lambda: _env_str("USER_AGENT"))
    # Needed when running headless Chromium as root inside many VPS/containers.
    no_sandbox: bool = field(default_factory=lambda: _env_bool("CRAWLER_NO_SANDBOX", False))
    chrome_extra_args: List[str] = field(default_factory=lambda: list(_DEFAULT_CHROME_ARGS))

    # --- Optional Redis-backed hash ring ---------------------------------- #
    redis_url: Optional[str] = field(default_factory=lambda: _env_str("REDIS_URL"))
    redis_namespace: str = field(default_factory=lambda: _env_str("REDIS_NAMESPACE", "sentry"))

    # --- Database (PostgreSQL + pgvector) --------------------------------- #
    # Leave empty to run in dry-run mode (no DB calls).
    db_dsn: Optional[str] = field(default_factory=lambda: _env_str("DB_DSN"))
    db_pool_max_size: int = field(default_factory=lambda: _env_int("DB_POOL_MAX_SIZE", 5))
    db_pool_min_size: int = field(default_factory=lambda: _env_int("DB_POOL_MIN_SIZE", 1))

    # --- Multi-Key LLM Pool ----------------------------------------------- #
    # Comma-separated list of Gemini keys. Falls back to gemini_api_key.
    gemini_api_keys_raw: Optional[str] = field(
        default_factory=lambda: _env_str("GEMINI_API_KEYS")
    )
    groq_api_key: Optional[str] = field(default_factory=lambda: _env_str("GROQ_API_KEY"))
    groq_model: str = field(
        default_factory=lambda: _env_str("GROQ_MODEL", "llama-3.1-70b-versatile")
    )
    # Safety margin: treat X% of provider RPM as the ceiling to avoid 429s.
    llm_rpm_safety_margin: float = field(
        default_factory=lambda: _env_float("LLM_RPM_SAFETY_MARGIN", 0.80)
    )
    # Gemini Flash free-tier RPM (2.5-flash).
    gemini_rpm_limit: int = field(default_factory=lambda: _env_int("GEMINI_RPM_LIMIT", 15))
    # Groq free-tier RPM.
    groq_rpm_limit: int = field(default_factory=lambda: _env_int("GROQ_RPM_LIMIT", 30))

    # --- FX / Arbitrage --------------------------------------------------- #
    # Open Exchange Rates (free tier) or CBSL endpoint.
    fx_api_url: str = field(
        default_factory=lambda: _env_str(
            "FX_API_URL",
            "https://open.er-api.com/v6/latest/USD",
        )
    )
    import_duty_factor: float = field(
        default_factory=lambda: _env_float("IMPORT_DUTY_FACTOR", 1.075)
    )

    # --- Forensic phase timeouts (seconds) -------------------------------- #
    forensic_phase1_timeout_s: int = field(
        default_factory=lambda: _env_int("FORENSIC_PHASE1_TIMEOUT_S", 60)
    )
    forensic_phase2_timeout_s: int = field(
        default_factory=lambda: _env_int("FORENSIC_PHASE2_TIMEOUT_S", 60)
    )
    forensic_phase3_timeout_s: int = field(
        default_factory=lambda: _env_int("FORENSIC_PHASE3_TIMEOUT_S", 90)
    )
    forensic_phase4_timeout_s: int = field(
        default_factory=lambda: _env_int("FORENSIC_PHASE4_TIMEOUT_S", 60)
    )
    forensic_phase5_timeout_s: int = field(
        default_factory=lambda: _env_int("FORENSIC_PHASE5_TIMEOUT_S", 30)
    )

    # --- YouTube Data API (optional, Phase 3) ----------------------------- #
    youtube_api_key: Optional[str] = field(
        default_factory=lambda: _env_str("YOUTUBE_API_KEY")
    )

    # ---------------------------------------------------------------------- #
    @property
    def sentry_char_budget(self) -> int:
        """Approximate character budget for the Sentry's LLM input slice."""
        return max(500, self.sentry_token_budget * self.chars_per_token)

    @property
    def gemini_api_keys(self) -> List[str]:
        """Return all Gemini API keys as a list (deduplicated, non-empty)."""
        raw = self.gemini_api_keys_raw or ""
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        # Fall back to the single-key field for backward compatibility.
        if not keys and self.gemini_api_key:
            keys = [self.gemini_api_key]
        return list(dict.fromkeys(keys))  # preserve order, deduplicate

    def browser_args(self) -> List[str]:
        args = list(self.chrome_extra_args)
        if self.no_sandbox:
            args += ["--no-sandbox", "--disable-setuid-sandbox"]
        return args

    def require_api_key(self) -> str:
        """Return the Gemini key or raise a clear, actionable error."""
        if not self.gemini_api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "in your environment or a .env file. Get one at "
                "https://aistudio.google.com/apikey"
            )
        return self.gemini_api_key


# Module-level singleton used across the app.
settings = Settings()
