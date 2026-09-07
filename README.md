# Smart Sentry — Anti-Fragile Scraping & Validation Engine

An autonomous, self-healing e-commerce scraper built on **Crawl4AI** + **Gemini
Flash** (`google-genai`). It doesn't just download pages — it classifies each
page's *state* in real time, avoids bot traps and cyclic loops, and emits only
strictly validated product data.

## How it works

Every URL flows through a gated pipeline:

```
 URL ─▶ [RING] dedup canonical URL
     ─▶ [FETCH] Crawl4AI (headless, images/CSS/fonts/media blocked)
     ─▶ [HASH] dedup content MD5  (cyclic-redirect / mirror guard)
     ─▶ [SENTRY] rule pre-flight ─▶ (if unclear) Gemini Flash classifier
     ─▶ branch:
          BOT_CHALLENGE_CAPTCHA  ─▶ [STEALTH] re-fetch with evasions, re-classify
          CATEGORY_GRID          ─▶ [QUEUE] enqueue child product links (BFS)
          PRODUCT_PAGE / OOS     ─▶ [EXTRACT] Gemini structured output ─▶ validated item
          ERROR_404 / UNKNOWN    ─▶ reject
```

## Files

| File               | Responsibility |
|--------------------|----------------|
| `config.py`        | Env-driven settings: keys, proxy, thresholds, memory flags |
| `schemas.py`       | Pydantic models: `PageInspectionResult`, `ScrapedProductItem` (+ strict validators) |
| `llm.py`           | Shared async `google-genai` wrapper with retries & structured output |
| `sentry.py`        | Rule-based pre-flight + Gemini Flash classification gate |
| `crawler.py`       | Crawl4AI orchestration, hash ring, stealth fallback, extraction, bounded BFS |
| `logging_utils.py` | Colored, gate-aware terminal logging |
| `main.py`          | End-to-end demo runner |

## Setup

```bash
pip install -r requirements.txt
crawl4ai-setup                 # installs the Playwright browser Crawl4AI drives
cp .env.example .env           # then add your GEMINI_API_KEY
```

Get a Gemini key at <https://aistudio.google.com/apikey> (`GOOGLE_API_KEY` also works).

## Run

```bash
python main.py                             # built-in demo (books.toscrape.com sandbox)
python main.py https://store.lk/p/123 ...  # your own targets
```

## Anti-fragility & memory (4 GB VPS)

- **Loop detection** — visited canonical URLs + content MD5 hashes (in-memory,
  or Redis via `REDIS_URL`).
- **Memory ceiling** — max 2 tabs, `text_mode`/`light_mode`, images/CSS/fonts/media
  aborted at the network layer, contexts closed + `gc.collect()` after each batch,
  V8 heap capped per renderer.
- **Bounded crawl** — `MAX_PAGES`, `MAX_DEPTH`, and per-category link caps stop a
  hostile site from making the crawl grow without limit.
- **Strict validation** — an item is emitted only if `price_lkr` is a finite,
  positive number; otherwise the page is kept as classification-only.

All thresholds are configurable — see `.env.example`.

> **Note:** the demo site prices are in GBP; the numeric value is still parsed
> into `price_lkr` (the field name assumes a Sri Lankan store). Respect each
> site's Terms of Service and `robots.txt` before scraping.
