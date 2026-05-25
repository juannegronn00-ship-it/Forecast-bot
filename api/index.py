from datetime import datetime, date, timedelta
from http.server import BaseHTTPRequestHandler
import sys
import os
import json
import logging
import traceback
import base64
import requests as _http

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Dedup state persisted in the GitHub repo as state/last_sent.txt.
# Requires GITHUB_TOKEN (PAT with repo scope) and GITHUB_REPO (owner/repo).
# Falls back to /tmp for local dev when env vars are absent.
_STATE_FILE = "state/last_sent.txt"
_cached_sha: str = ""  # populated by _gh_read, consumed by _gh_write to avoid extra API call


def _gh_headers() -> dict:
    return {
        "Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}",
        "Accept": "application/vnd.github.v3+json",
    }


def _gh_contents_url() -> str:
    repo = os.getenv("GITHUB_REPO", "")
    return f"https://api.github.com/repos/{repo}/contents/{_STATE_FILE}"


def _gh_read_last_sent() -> str:
    """Return the stored date string (e.g. '2026-05-05') or '' on any failure."""
    global _cached_sha
    if not os.getenv("GITHUB_TOKEN") or not os.getenv("GITHUB_REPO"):
        return ""
    try:
        resp = _http.get(_gh_contents_url(), headers=_gh_headers(), timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            _cached_sha = data.get("sha", "")
            return base64.b64decode(data["content"]).decode().strip()
        _cached_sha = ""
        return ""
    except Exception as e:
        logger.warning(f"GitHub dedup read failed: {e}")
        return ""


def _gh_write_last_sent(date_str: str) -> bool:
    """Write date_str to state file via GitHub Contents API. Returns True on success."""
    if not os.getenv("GITHUB_TOKEN") or not os.getenv("GITHUB_REPO"):
        return False
    try:
        body: dict = {
            "message": f"chore: dedup {date_str} [skip ci]",
            "content": base64.b64encode(date_str.encode()).decode(),
            "committer": {"name": "forecast-bot", "email": "bot@forecast.local"},
        }
        if _cached_sha:
            body["sha"] = _cached_sha  # required to update an existing file
        resp = _http.put(_gh_contents_url(), headers=_gh_headers(), json=body, timeout=10)
        return resp.status_code in (200, 201)
    except Exception as e:
        logger.warning(f"GitHub dedup write failed: {e}")
        return False


# ── Valid send window (UTC hours, inclusive start / exclusive end) ────────────
# The Vercel cron fires at 0 11 * * * (11:00 UTC = ~7 AM ET).
# Any invocation outside this window is a rogue run: it is rejected before the
# pipeline starts, will NOT send, and will NOT write to sent_forecasts, so it
# cannot block the legitimate morning cron.
# Window 09:00–14:59 UTC covers 5 AM–10 AM ET across all DST transitions.
_VALID_SEND_WINDOW_UTC_START: int = 9
_VALID_SEND_WINDOW_UTC_END:   int = 15  # exclusive


def _is_valid_send_window() -> bool:
    """Return True if the current UTC hour is inside the legitimate send window."""
    return _VALID_SEND_WINDOW_UTC_START <= datetime.utcnow().hour < _VALID_SEND_WINDOW_UTC_END


def _forecast_target_date() -> str:
    """
    Return the DA forecast target date (always tomorrow) as an ISO string.

    Using tomorrow as the idempotency key means a rogue run at 2 AM UTC on
    May 25 keys on '2026-05-26', while the scheduled 11 AM UTC run on May 24
    keys on '2026-05-25' — different dates, no collision across calendar days.
    """
    return (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")


def _authorized(headers) -> bool:
    secret = os.getenv("CRON_SECRET", "")
    if not secret:
        return True
    return headers.get("Authorization", "") == f"Bearer {secret}"


def _already_sent_for_tomorrow() -> bool:
    """
    Check whether the forecast for tomorrow has already been sent.

    Delegates to already_sent_for() which RAISES RuntimeError on any Supabase
    configuration or network error — this function intentionally lets that
    exception propagate so the caller can abort (fails closed).
    """
    target = _forecast_target_date()
    from src.db import supabase_client as _db
    # already_sent_for() raises on error — do NOT swallow here; let it propagate
    if _db.already_sent_for(target):
        logger.info(f"Supabase dedup: forecast for {target} already sent")
        return True
    # Secondary: GitHub Contents API (belt-and-suspenders for cross-instance state)
    if _gh_read_last_sent() == target:
        logger.info(f"GitHub dedup: forecast for {target} already sent")
        return True
    return False


def _mark_sent_today() -> None:
    """
    Record that today's forecast (for tomorrow) was successfully delivered.

    Uses tomorrow as the key so rogue off-hours runs that didn't go through
    the time-window gate cannot pollute the idempotency table.
    """
    target = _forecast_target_date()
    # Primary: Supabase sent_forecasts (idempotent INSERT, ignores duplicates)
    try:
        from src.db import supabase_client as _db
        if _db.mark_sent_for(target):
            logger.info(f"Supabase dedup: recorded sent for target={target}")
    except Exception as e:
        logger.warning(f"Supabase mark_sent_for failed: {e}")
    # Secondary: GitHub Contents API
    if _gh_write_last_sent(target):
        logger.info(f"GitHub dedup state written: {target}")
        return
    # Tertiary: /tmp flag (local-dev only; ephemeral on Vercel)
    try:
        with open(f"/tmp/forecast_sent_{target}.flag", "w") as f:
            f.write(datetime.utcnow().isoformat())
    except Exception:
        pass


def _build_status() -> dict:
    """
    Assemble the /status payload: last run, scraper health, 7-day accuracy.
    Non-fatal — returns partial data if Supabase is unavailable.
    """
    from src.db import supabase_client as db

    # Last successful forecast
    forecasts = db.select("forecasts", "", limit=1)
    last_forecast = forecasts[0] if forecasts else None

    # Scraper health from last run
    health_rows = db.select("scraper_health", "", limit=20)
    last_run_date = health_rows[0].get("run_date") if health_rows else None
    scraper_status: dict = {}
    if last_run_date:
        for row in health_rows:
            if row.get("run_date") == last_run_date:
                scraper_status[row["scraper_name"]] = row["success"]

    # 7-day accuracy
    errors = db.select("forecast_errors", "", limit=7)
    maes   = [r.get("mae_total") for r in errors if r.get("mae_total") is not None]
    block_17_24 = [r.get("mae_block_17_24") for r in errors if r.get("mae_block_17_24") is not None]

    return {
        "last_run_date":     last_run_date,
        "last_forecast_date": last_forecast.get("forecast_date") if last_forecast else None,
        "scraper_health":    scraper_status,
        "accuracy_7d": {
            "n_days":         len(maes),
            "mae_avg_total":  round(sum(maes) / len(maes), 2) if maes else None,
            "mae_avg_17_24":  round(sum(block_17_24) / len(block_17_24), 2) if block_17_24 else None,
        },
    }


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if not _authorized(self.headers):
            self._respond(401, {"error": "Unauthorized"})
            return

        # /status endpoint — returns observability data without running the pipeline
        if "status" in (self.path or ""):
            try:
                self._respond(200, _build_status())
            except Exception as e:
                self._respond(500, {"error": str(e)})
            return

        # ── BUG 1 FIX: time-window gate ──────────────────────────────────────
        # Reject any invocation outside the valid UTC send window so rogue runs
        # (Vercel retries, webhook triggers, accidental hits) cannot run the
        # pipeline, send a forecast, or write to sent_forecasts.
        if not _is_valid_send_window():
            utc_hour = datetime.utcnow().hour
            logger.info(
                f"Outside valid send window ({_VALID_SEND_WINDOW_UTC_START}–"
                f"{_VALID_SEND_WINDOW_UTC_END} UTC, current={utc_hour} UTC) — skipping."
            )
            self._respond(200, {
                "skipped":    True,
                "reason":     "outside_send_window",
                "utc_hour":   utc_hour,
                "window_utc": f"{_VALID_SEND_WINDOW_UTC_START}:00–{_VALID_SEND_WINDOW_UTC_END}:00",
            })
            return

        # ── BUG 2 FIX: dedup check — fails CLOSED on any DB error ────────────
        # _already_sent_for_tomorrow() raises RuntimeError if Supabase is
        # unreachable or misconfigured. We catch it here and return 500 so the
        # pipeline never runs on an unknown dedup state.
        try:
            if _already_sent_for_tomorrow():
                target = _forecast_target_date()
                logger.info(f"Forecast for {target} already sent — skipping.")
                self._respond(200, {"skipped": True, "reason": "already_sent", "target": target})
                return
        except RuntimeError as e:
            logger.error(f"Dedup check failed (failing closed — will not send): {e}")
            self._respond(500, {
                "success": False,
                "error":   f"Dedup check failed: {e}",
                "reason":  "supabase_dedup_error",
            })
            return

        logger.info("=" * 60)
        logger.info(f"CRON TRIGGERED at {datetime.utcnow().isoformat()} UTC  pid={os.getpid()}")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%B %d, %Y")
        logger.info(f"Generating forecast for: {tomorrow}")
        logger.info("=" * 60)

        try:
            from src.main import ForecastBot
            from src.actuals_logger import run_daily_actuals_check
            from src.weekly_digest import is_monday, send_weekly_digest

            # ── Step 0a: Compute yesterday's forecast errors (feedback loop) ─
            logger.info("Running daily actuals check...")
            actuals_result = run_daily_actuals_check()
            if actuals_result:
                logger.info(
                    f"✅ Actuals check: yesterday MAE=${actuals_result['mae_total']:.2f}"
                )
            else:
                logger.info("Actuals check: no data (first run or unavailable)")

            # ── Step 0b: Monday weekly digest ─────────────────────────────
            if is_monday():
                logger.info("Monday — sending weekly accuracy digest...")
                # We need the sender; create a minimal instance for this
                from src.messengers.telegram_sender import TelegramSender
                _sender = TelegramSender()
                send_weekly_digest(_sender)


            bot = ForecastBot()

            final_prices = bot.run_forecast()

            if not final_prices or len(final_prices) != 24:
                msg = f"Pipeline returned {len(final_prices) if final_prices else 0} prices (expected 24)"
                logger.error(f"❌ {msg}")
                self._respond(500, {"success": False, "error": msg})
                return

            avg = sum(final_prices) / len(final_prices)
            logger.info(f"Pipeline complete | avg=${avg:.2f} | range=${min(final_prices):.2f}–${max(final_prices):.2f}")

            logger.info(f"📤 SEND START at {datetime.utcnow().isoformat()} UTC")
            tg_result = bot.send_telegram(final_prices)
            logger.info(f"📤 SEND END   at {datetime.utcnow().isoformat()} UTC")
            stepdad_ok = tg_result.get("stepdad_ok", False)
            you_ok = tg_result.get("you_ok", False)
            tg_errors = tg_result.get("errors", [])

            # Mark sent here too — covers the API path (send_telegram marks it
            # in the local-run path; both paths write the same flag file)
            if stepdad_ok:
                _mark_sent_today()
                logger.info(f"✅ Telegram delivered to stepdad for {tomorrow}")
            else:
                logger.error(f"❌ Telegram to stepdad FAILED. Errors: {tg_errors}")

            self._respond(200, {
                "success": True,
                "forecast_date": tomorrow,
                "telegram": {
                    "stepdad": stepdad_ok,
                    "monitor": you_ok,
                    "errors": tg_errors,
                },
                "forecast": {
                    "hours": final_prices,
                    "avg": round(avg, 2),
                    "min": round(min(final_prices), 2),
                    "max": round(max(final_prices), 2),
                },
            })

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"❌ PIPELINE EXCEPTION: {e}\n{tb}")
            self._respond(500, {"success": False, "error": str(e), "traceback": tb})

    def do_POST(self):
        self.do_GET()

    def _respond(self, status: int, body: dict):
        payload = json.dumps(body, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        # Route BaseHTTPRequestHandler logs through our logger
        logger.info(fmt % args)
