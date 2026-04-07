# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Set up the Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the bot (runs one forecast immediately, then starts daily scheduler)
python -m src.main

# Run a single forecast without the scheduler (for testing)
python -c "from src.main import ForecastBot; ForecastBot().run_once()"
```

## Architecture

This is a DA-LMP (Day-Ahead Locational Marginal Price) energy price forecast bot for the MISO market. It runs a 6-step pipeline daily at 7:00 AM, delivering 24-hour price forecasts via WhatsApp.

**Pipeline flow** (`src/main.py` — `ForecastBot.run_forecast`):
1. **Scrape** — `FantodsScraperr` (primary) fetches tomorrow's DA-LMP prices + load/wind forecasts from a third-party table. `MISOScraper` fetches MISO official CSVs as validation only; failures are non-fatal.
2. **Base prices** — Currently uses fantods prices directly. `HourMatcher` (in `src/utils/matcher.py`) contains the full similar-day matching algorithm (weighted 70% load / 30% wind) but it's not wired into the main pipeline yet — it requires historical data that isn't persisted.
3. **Claude refinement** — `AIRefiner.claude_refine` sends base prices + forecasts to `claude-3-5-sonnet-20241022`, which returns percentage adjustments per hour as JSON.
4. **Gemini validation** — `AIRefiner.gemini_validate` cross-checks Claude's prices with Gemini. If validation fails, final prices are averaged between base and Claude outputs. Gemini is currently commented out in `AIRefiner.__init__` (`genai.configure` line).
5. **Merge** — `AIRefiner.merge_results` picks Claude prices (Gemini approved) or blended prices (Gemini flagged).
6. **Send** — `WhatsAppSender` sends via Twilio to `STEPDAD_WHATSAPP` (primary recipient) and `YOUR_WHATSAPP` (monitoring).

## Environment Variables

Required in `.env`:
- `CLAUDE_API_KEY` — Anthropic API key (used by `Anthropic()` client which reads `ANTHROPIC_API_KEY` by default — note potential key name mismatch)
- `GEMINI_API_KEY` — Google Gemini API key
- `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` — Twilio credentials
- `STEPDAD_WHATSAPP` — recipient phone number (e.g. `+1XXXXXXXXXX`)
- `YOUR_WHATSAPP` — optional monitoring number

## Known Issues / MVP State

- `HourMatcher.generate_base_forecast` is fully implemented but unused — `run_forecast` bypasses it and uses fantods prices directly as base prices, with a synthetic fallback (`50.0 + load * 0.1`).
- The `Anthropic()` client reads `ANTHROPIC_API_KEY` from env by default, but `.env` sets `CLAUDE_API_KEY`. Either rename the env var or pass it explicitly: `Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))`.
- Gemini integration is disabled (`genai.configure` is commented out in `AIRefiner.__init__`), so Gemini validation always falls back to the exception handler.
