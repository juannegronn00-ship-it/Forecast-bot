-- DA-LMP Forecast Bot: observability and feedback loop tables
-- Run once in the Supabase SQL editor (or via psql)
-- All columns follow snake_case convention; dates use timestamptz

-- ── forecasts ─────────────────────────────────────────────────────────────
-- One row per forecast target date.
-- forecast_date = the delivery date being forecast (tomorrow at time of generation).
-- Prices and confidence are JSON objects keyed by hour string "1".."24".
CREATE TABLE IF NOT EXISTS forecasts (
    id                  BIGSERIAL PRIMARY KEY,
    forecast_date       DATE        NOT NULL,
    generated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prices              JSONB       NOT NULL,
    confidence          JSONB,
    market_bias         TEXT        CHECK (market_bias IN ('BULLISH', 'BEARISH', 'NEUTRAL')),
    market_bias_reason  TEXT,
    signal_summary      TEXT,
    peak_driver         TEXT,
    risk_flags          TEXT,
    base_prices         JSONB,
    scraper_health      JSONB,
    UNIQUE (forecast_date)
);

-- ── actuals ───────────────────────────────────────────────────────────────
-- Actual cleared DA-LMP prices for ILLINOIS.HUB, scraped from MISO ExAnte CSV.
-- delivery_date = the day the prices applied to (same as forecast_date above).
CREATE TABLE IF NOT EXISTS actuals (
    id              BIGSERIAL PRIMARY KEY,
    delivery_date   DATE        NOT NULL,
    scraped_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prices          JSONB       NOT NULL,
    source          TEXT        NOT NULL DEFAULT 'miso_exante',
    node            TEXT        NOT NULL DEFAULT 'ILLINOIS.HUB',
    UNIQUE (delivery_date)
);

-- ── forecast_errors ───────────────────────────────────────────────────────
-- Per-hour errors (forecast − actual) and MAE summary stats.
-- Positive error = forecast was too high; negative = too low.
-- Used to compute bias corrections and the weekly accuracy digest.
CREATE TABLE IF NOT EXISTS forecast_errors (
    id               BIGSERIAL PRIMARY KEY,
    delivery_date    DATE        NOT NULL,
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    errors           JSONB       NOT NULL,
    mae_total        FLOAT,
    mae_block_1_8    FLOAT,
    mae_block_9_16   FLOAT,
    mae_block_17_24  FLOAT,
    UNIQUE (delivery_date)
);

-- ── scraper_health ────────────────────────────────────────────────────────
-- Log of each scraper's success/failure per cron run.
-- Used to track data quality over time and power the DEGRADED DATA warning.
CREATE TABLE IF NOT EXISTS scraper_health (
    id              BIGSERIAL PRIMARY KEY,
    run_date        DATE        NOT NULL,
    logged_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scraper_name    TEXT        NOT NULL,
    success         BOOLEAN     NOT NULL,
    source          TEXT,
    error_message   TEXT,
    fallback_used   BOOLEAN     NOT NULL DEFAULT FALSE
);

-- ── Indexes ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_forecasts_date
    ON forecasts (forecast_date DESC);

CREATE INDEX IF NOT EXISTS idx_actuals_date
    ON actuals (delivery_date DESC);

CREATE INDEX IF NOT EXISTS idx_forecast_errors_date
    ON forecast_errors (delivery_date DESC);

CREATE INDEX IF NOT EXISTS idx_scraper_health_run
    ON scraper_health (run_date DESC, scraper_name);

-- ── sent_forecasts ───────────────────────────────────────────────────────
-- Idempotency table: one row per calendar day a forecast was successfully sent.
-- The UNIQUE constraint on forecast_date is the race-condition fence —
-- even if Vercel invokes the function multiple times concurrently, only the
-- first INSERT succeeds; all subsequent calls see an existing row and skip.
CREATE TABLE IF NOT EXISTS sent_forecasts (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    forecast_date DATE        NOT NULL UNIQUE,
    sent_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sent_forecasts_date
    ON sent_forecasts (forecast_date DESC);

-- ── Migration 2026-05-25: fix idempotency key semantics ──────────────────
-- Previously the code stored the *run date* (date.today()) in forecast_date.
-- A rogue off-hours invocation on the same UTC calendar day as the scheduled
-- cron could write the row first, causing the cron to skip.
--
-- Fix: the code now stores the DA *target date* (tomorrow = run_date + 1 day)
-- so rogue runs from the night before key on a different date than the
-- scheduled morning cron.
--
-- One-time cleanup: remove the stale row that was blocking the 2026-05-26
-- morning send, so the next legitimate invocation can proceed.
DELETE FROM sent_forecasts WHERE forecast_date = '2026-05-26';

-- ── Row-level security (optional) ────────────────────────────────────────
-- If you use the anon key (not service role), enable RLS and add insert policies:
--   ALTER TABLE forecasts ENABLE ROW LEVEL SECURITY;
--   CREATE POLICY "service-role only" ON forecasts USING (auth.role() = 'service_role');
-- (Repeat for actuals, forecast_errors, scraper_health, sent_forecasts)
-- With SUPABASE_SERVICE_ROLE_KEY, RLS can remain disabled for simplicity.
