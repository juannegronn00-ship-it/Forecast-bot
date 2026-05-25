"""
Lightweight Supabase REST client using requests (no external library).

Requires env vars:
  SUPABASE_URL              — e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY — service role key (bypasses RLS)
    OR
  SUPABASE_ANON_KEY         — anon key (if RLS policies allow)

All functions are non-fatal: they log warnings on failure and return
safe defaults so the forecast pipeline never crashes on DB issues.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL: str = ""
_KEY:      str = ""


def _init() -> bool:
    global _BASE_URL, _KEY
    if _BASE_URL and _KEY:
        return True
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
        or os.getenv("SUPABASE_ANON_KEY", "")
    )
    if not url or not key:
        return False
    _BASE_URL = url
    _KEY = key
    return True


def _headers(prefer: str = "return=minimal") -> Dict[str, str]:
    return {
        "apikey":        _KEY,
        "Authorization": f"Bearer {_KEY}",
        "Content-Type":  "application/json",
        "Prefer":        prefer,
    }


def upsert(table: str, data: Dict[str, Any]) -> bool:
    """Insert or update a row (ON CONFLICT DO UPDATE). Returns True on success."""
    if not _init():
        logger.debug(f"Supabase not configured — skipping upsert({table})")
        return False
    try:
        resp = requests.post(
            f"{_BASE_URL}/rest/v1/{table}",
            headers=_headers("resolution=merge-duplicates,return=minimal"),
            json=data,
            timeout=8,
        )
        if resp.status_code in (200, 201):
            return True
        logger.warning(f"Supabase upsert({table}): HTTP {resp.status_code} — {resp.text[:200]}")
        return False
    except Exception as e:
        logger.warning(f"Supabase upsert({table}) failed: {e}")
        return False


def insert_many(table: str, rows: List[Dict[str, Any]]) -> bool:
    """Insert multiple rows without conflict resolution."""
    if not rows:
        return True
    if not _init():
        return False
    try:
        resp = requests.post(
            f"{_BASE_URL}/rest/v1/{table}",
            headers=_headers(),
            json=rows,
            timeout=8,
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        logger.warning(f"Supabase insert_many({table}) failed: {e}")
        return False


def select(
    table: str,
    filters: str = "",
    order:  str = "id.desc",
    limit:  int = 10,
) -> List[Dict[str, Any]]:
    """
    Select rows from table. `filters` is a raw PostgREST filter string,
    e.g. 'delivery_date=eq.2026-05-18'.
    """
    if not _init():
        return []
    try:
        params_str = f"{filters}&" if filters else ""
        url = f"{_BASE_URL}/rest/v1/{table}?{params_str}order={order}&limit={limit}"
        resp = requests.get(url, headers=_headers(""), timeout=8)
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Supabase select({table}): HTTP {resp.status_code}")
        return []
    except Exception as e:
        logger.warning(f"Supabase select({table}) failed: {e}")
        return []


def already_sent_today(forecast_date: str) -> bool:
    """
    Deprecated alias — use already_sent_for() for strict (fail-closed) checking.
    Returns False on error instead of raising, preserved for backward compat.
    """
    try:
        return already_sent_for(forecast_date)
    except RuntimeError:
        return False


def already_sent_for(forecast_for_date: str) -> bool:
    """
    Return True if a successful send is already recorded for *forecast_for_date*
    (ISO string — always the DA *target* date, i.e. tomorrow at run time).

    RAISES RuntimeError on any configuration or network error so the caller
    can abort cleanly instead of silently proceeding to send (fails closed).
    """
    if not _init():
        raise RuntimeError(
            "Supabase not configured — SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY missing"
        )
    try:
        resp = requests.get(
            f"{_BASE_URL}/rest/v1/sent_forecasts"
            f"?forecast_date=eq.{forecast_for_date}&select=id&limit=1",
            headers=_headers(""),
            timeout=8,
        )
    except Exception as e:
        raise RuntimeError(f"Supabase dedup check network error: {e}") from e

    if resp.status_code == 200:
        return len(resp.json()) > 0
    raise RuntimeError(
        f"Supabase dedup check returned HTTP {resp.status_code}: {resp.text[:200]}"
    )


def mark_sent_today(forecast_date: str) -> bool:
    """Deprecated alias for mark_sent_for(). Kept for backward compatibility."""
    return mark_sent_for(forecast_date)


def mark_sent_for(forecast_for_date: str) -> bool:
    """
    Record that the forecast for *forecast_for_date* was successfully delivered.
    *forecast_for_date* is the DA target date (always tomorrow at run time).

    Uses `resolution=ignore-duplicates` so concurrent Vercel invocations are
    safe: the UNIQUE constraint means only the first INSERT wins; subsequent
    calls are silently ignored and return True.

    Returns True on success or conflict, False on hard error (non-fatal write).
    """
    if not _init():
        logger.debug("Supabase not configured — skipping mark_sent_for")
        return False
    try:
        resp = requests.post(
            f"{_BASE_URL}/rest/v1/sent_forecasts",
            headers=_headers("resolution=ignore-duplicates,return=minimal"),
            json={"forecast_date": forecast_for_date},
            timeout=8,
        )
        # 201 = inserted; 200 = conflict ignored — both are success
        if resp.status_code in (200, 201):
            return True
        logger.warning(
            f"Supabase mark_sent_for({forecast_for_date}): "
            f"HTTP {resp.status_code} — {resp.text[:200]}"
        )
        return False
    except Exception as e:
        logger.warning(f"Supabase mark_sent_for failed: {e}")
        return False


def is_configured() -> bool:
    return _init()
