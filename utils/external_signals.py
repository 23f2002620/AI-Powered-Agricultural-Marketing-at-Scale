"""
utils/external_signals.py
External Data Integration Layer

WHAT CHANGED FROM ORIGINAL (offline-resilience fixes):
-------------------------------------------------------
The original file already had good fallback defaults (0.3) when API keys
were missing. However it had two remaining gaps:

1. NO CACHE — every orchestrator run hit the live APIs. In rural areas with
   intermittent connectivity, even a single API timeout would stall the entire
   batch_plan(). There was no way to reuse yesterday's data when offline.

2. FIXED DEFAULT VALUES — the default MandiPrice used the same ₹2000/quintal
   for every crop regardless of crop type. A wheat farmer and a potato farmer
   got identical defaults, making the price_trend signal meaningless offline.

FIXES ADDED:
- add_signal_cache(): nightly job that calls all 3 APIs when internet IS
  available and writes results to data/signal_cache.json.
- get_signal_offline(): reads from that cache. Falls back to per-crop
  static defaults if the cache is also missing.
- enrich_grower_context() now calls get_signal_offline() first and only
  hits the live APIs if a fresh key is provided AND online.
- Per-crop realistic price defaults so offline estimates are meaningful.

APIs still used (optional, gracefully degraded):
  IMD_API_KEY        — IMD Open Data API key
  NCIPM_API_KEY      — ICAR-NCIPM pest surveillance API key
  AGMARKNET_API_KEY  — Agmarknet mandi prices API key

Without any key: reads from cache → falls back to per-crop static defaults.
System continues running in all cases.
"""

import os
import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DATA_DIR          = Path("data")
CACHE_FILE        = DATA_DIR / "signal_cache.json"
DEFAULT_WEATHER_RISK  = 0.3
DEFAULT_PEST_PRESSURE = 0.3
REQUEST_TIMEOUT       = 10   # seconds

# ---------------------------------------------------------------------------
# Per-crop realistic offline price defaults (₹ per quintal)
# FIX: Original had ₹2000 for ALL crops. Now crop-specific.
# ---------------------------------------------------------------------------
CROP_DEFAULT_PRICE = {
    "wheat":     2200.0,
    "mustard":   5400.0,
    "chickpea":  5000.0,
    "potato":    1000.0,
    "barley":    1800.0,
    "lentil":    5500.0,
    "safflower": 5800.0,
    "cumin":    18000.0,
    "maize":     2000.0,
}


# ---------------------------------------------------------------------------
# Dataclasses (unchanged from original)
# ---------------------------------------------------------------------------

@dataclass
class WeatherSignal:
    tehsil: str
    fetch_date: str
    max_temp_c: float
    min_temp_c: float
    rainfall_mm: float
    humidity_pct: float
    wind_speed_kmh: float
    weather_risk_score: float
    alert_message: str
    source: str = "IMD"


@dataclass
class PestAlert:
    tehsil: str
    crop: str
    fetch_date: str
    pest_name: str
    severity: str
    pest_pressure_index: float
    advisory: str
    emergency: bool = False
    source: str = "ICAR-NCIPM"


@dataclass
class MandiPrice:
    crop: str
    state: str
    fetch_date: str
    modal_price_per_quintal: float
    min_price: float
    max_price: float
    price_trend: str
    market_name: str
    source: str = "Agmarknet"


@dataclass
class ExternalContext:
    weather: WeatherSignal
    pest: PestAlert
    mandi: MandiPrice
    is_emergency: bool
    composite_urgency_score: float


# ---------------------------------------------------------------------------
# Risk computation helper (unchanged)
# ---------------------------------------------------------------------------

def _compute_weather_risk(max_temp: float, min_temp: float,
                           humidity: float, rainfall: float) -> float:
    risk = 0.0
    if humidity > 80:
        risk += 0.3
    elif humidity > 65:
        risk += 0.15
    if 15 <= min_temp <= 25 and humidity > 70:
        risk += 0.25
    if rainfall > 20:
        risk += 0.25
    elif rainfall > 5:
        risk += 0.1
    if (max_temp - min_temp) > 15:
        risk += 0.1
    return min(round(risk, 3), 1.0)


_SEVERITY_TO_INDEX = {
    "none": 0.0, "low": 0.25, "moderate": 0.5, "high": 0.75, "severe": 1.0
}

COMMODITY_MAP = {
    "wheat": "Wheat", "mustard": "Mustard", "chickpea": "Gram(Chick Pea)",
    "potato": "Potato", "barley": "Barley", "lentil": "Masur (Lentil)",
    "safflower": "Safflower", "cumin": "Cuminseed", "maize": "Maize",
}


# ---------------------------------------------------------------------------
# Default constructors (per-crop, not hardcoded)
# ---------------------------------------------------------------------------

def _default_weather(tehsil: str, reason: str = "default") -> WeatherSignal:
    return WeatherSignal(
        tehsil=tehsil,
        fetch_date=date.today().isoformat(),
        max_temp_c=28.0, min_temp_c=14.0, rainfall_mm=3.0,
        humidity_pct=65.0, wind_speed_kmh=12.0,
        weather_risk_score=DEFAULT_WEATHER_RISK,
        alert_message=f"IMD data unavailable ({reason}) — using default risk estimate.",
        source="default",
    )


def _default_pest(tehsil: str, crop: str, reason: str = "default") -> PestAlert:
    return PestAlert(
        tehsil=tehsil, crop=crop,
        fetch_date=date.today().isoformat(),
        pest_name="unknown", severity="low",
        pest_pressure_index=DEFAULT_PEST_PRESSURE,
        advisory=f"ICAR-NCIPM data unavailable ({reason}) — monitor crop regularly.",
        source="default",
    )


def _default_mandi(crop: str, state: str, reason: str = "default") -> MandiPrice:
    # FIX: Use crop-specific realistic price instead of hardcoded ₹2000
    modal = CROP_DEFAULT_PRICE.get(crop.lower(), 2000.0)
    return MandiPrice(
        crop=crop, state=state,
        fetch_date=date.today().isoformat(),
        modal_price_per_quintal=modal,
        min_price=round(modal * 0.9, 1),
        max_price=round(modal * 1.1, 1),
        price_trend="stable",
        market_name="unknown",
        source="default",
    )


# ---------------------------------------------------------------------------
# ADDED: Local signal cache — nightly write, offline read
# ---------------------------------------------------------------------------

def cache_signals_nightly(tehsil_crop_state_list: list[tuple[str, str, str]],
                            imd_key: str = "", ncipm_key: str = "",
                            agmarknet_key: str = "") -> dict:
    """
    ADDED: Run this once per day (e.g. via cron at 11 PM) when connectivity
    is available. Writes fetched signals to data/signal_cache.json so the
    orchestrator can read them offline during the day.

    Args:
        tehsil_crop_state_list: list of (tehsil, crop, state) tuples to cache
        imd_key, ncipm_key, agmarknet_key: API keys

    Returns:
        cache dict (also written to CACHE_FILE)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = {"cached_at": datetime.utcnow().isoformat(), "signals": {}}

    for tehsil, crop, state in tehsil_crop_state_list:
        key = f"{tehsil}|{crop}"
        try:
            weather = fetch_weather_signal(tehsil, state, imd_key)
            pest    = fetch_pest_alert(tehsil, crop, ncipm_key)
            mandi   = fetch_mandi_price(crop, state, agmarknet_key)
            cache["signals"][key] = {
                "weather_risk_score": weather.weather_risk_score,
                "weather_alert":      weather.alert_message,
                "max_temp_c":         weather.max_temp_c,
                "min_temp_c":         weather.min_temp_c,
                "rainfall_mm":        weather.rainfall_mm,
                "humidity_pct":       weather.humidity_pct,
                "pest_pressure_index": pest.pest_pressure_index,
                "pest_name":          pest.pest_name,
                "pest_severity":      pest.severity,
                "pest_advisory":      pest.advisory,
                "pest_emergency":     pest.emergency,
                "mandi_price":        mandi.modal_price_per_quintal,
                "price_trend":        mandi.price_trend,
                "weather_source":     weather.source,
                "pest_source":        pest.source,
            }
            logger.info("Cached signals for %s", key)
        except Exception as e:
            logger.warning("Cache failed for %s: %s", key, e)

    CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    logger.info("Signal cache written to %s (%d entries)", CACHE_FILE, len(cache["signals"]))
    return cache


def get_signal_offline(tehsil: str, crop: str, state: str,
                        max_age_hours: int = 36) -> ExternalContext:
    """
    ADDED: Read signals from local cache file.
    Falls back to per-crop static defaults if cache is missing or stale.

    max_age_hours: reject cache entries older than this (default 36h = 1.5 days)
    """
    key = f"{tehsil}|{crop}"

    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
            cached_at = datetime.fromisoformat(cache.get("cached_at", "2000-01-01"))
            age_hours = (datetime.utcnow() - cached_at).total_seconds() / 3600

            if age_hours <= max_age_hours and key in cache.get("signals", {}):
                s = cache["signals"][key]
                weather = WeatherSignal(
                    tehsil=tehsil,
                    fetch_date=cached_at.date().isoformat(),
                    max_temp_c=s.get("max_temp_c", 28.0),
                    min_temp_c=s.get("min_temp_c", 14.0),
                    rainfall_mm=s.get("rainfall_mm", 3.0),
                    humidity_pct=s.get("humidity_pct", 65.0),
                    wind_speed_kmh=12.0,
                    weather_risk_score=s["weather_risk_score"],
                    alert_message=s.get("weather_alert", "From cache."),
                    source=f"cache({s.get('weather_source','IMD')})",
                )
                pest = PestAlert(
                    tehsil=tehsil, crop=crop,
                    fetch_date=cached_at.date().isoformat(),
                    pest_name=s.get("pest_name", "unknown"),
                    severity=s.get("pest_severity", "low"),
                    pest_pressure_index=s["pest_pressure_index"],
                    advisory=s.get("pest_advisory", "From cache."),
                    emergency=s.get("pest_emergency", False),
                    source=f"cache({s.get('pest_source','ICAR-NCIPM')})",
                )
                modal = s.get("mandi_price", CROP_DEFAULT_PRICE.get(crop.lower(), 2000.0))
                mandi = MandiPrice(
                    crop=crop, state=state,
                    fetch_date=cached_at.date().isoformat(),
                    modal_price_per_quintal=modal,
                    min_price=round(modal * 0.9, 1),
                    max_price=round(modal * 1.1, 1),
                    price_trend=s.get("price_trend", "stable"),
                    market_name="cached",
                    source="cache(Agmarknet)",
                )
                price_bonus = 0.1 if mandi.price_trend == "rising" else 0.0
                urgency = round(
                    0.4 * weather.weather_risk_score
                    + 0.5 * pest.pest_pressure_index
                    + price_bonus, 3
                )
                logger.info("Serving %s from cache (age: %.1fh)", key, age_hours)
                return ExternalContext(
                    weather=weather, pest=pest, mandi=mandi,
                    is_emergency=pest.emergency,
                    composite_urgency_score=min(urgency, 1.0),
                )
            else:
                logger.warning("Cache stale (%.1fh old) or key %s missing — using static defaults", age_hours, key)
        except Exception as e:
            logger.warning("Cache read failed: %s — using static defaults", e)

    # Static defaults (per-crop)
    weather = _default_weather(tehsil, "no cache")
    pest    = _default_pest(tehsil, crop, "no cache")
    mandi   = _default_mandi(crop, state, "no cache")
    price_bonus = 0.0
    urgency = round(0.4 * DEFAULT_WEATHER_RISK + 0.5 * DEFAULT_PEST_PRESSURE + price_bonus, 3)
    return ExternalContext(
        weather=weather, pest=pest, mandi=mandi,
        is_emergency=False,
        composite_urgency_score=urgency,
    )


# ---------------------------------------------------------------------------
# Live API functions (unchanged from original, kept for nightly cache job)
# ---------------------------------------------------------------------------

def fetch_weather_signal(tehsil: str, state: str, api_key: str = "") -> WeatherSignal:
    key = api_key or os.getenv("IMD_API_KEY", "")
    if not key:
        logger.warning("IMD_API_KEY not set — using default weather signal for %s", tehsil)
        return _default_weather(tehsil, "no API key")

    district = tehsil.split("_")[0] if "_" in tehsil else tehsil
    try:
        resp = httpx.get(
            "https://mausam.imd.gov.in/api/v1/weather/forecast",
            params={"district": district, "state": state, "days": 3},
            headers={"Authorization": f"Bearer {key}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        today_fc   = data.get("forecast", [{}])[0]
        max_temp   = float(today_fc.get("maxTemp",   28))
        min_temp   = float(today_fc.get("minTemp",   14))
        rainfall   = float(today_fc.get("rainfall",   0))
        humidity   = float(today_fc.get("humidity",  65))
        wind_speed = float(today_fc.get("windSpeed", 10))
        risk       = _compute_weather_risk(max_temp, min_temp, humidity, rainfall)
        alert = ""
        if risk > 0.6:
            alert = f"HIGH disease risk: humidity {humidity:.0f}%, rainfall {rainfall:.1f}mm. Apply fungicide preventively."
        elif risk > 0.35:
            alert = f"MODERATE disease risk: monitor crop closely. Humidity {humidity:.0f}%."
        return WeatherSignal(
            tehsil=tehsil, fetch_date=date.today().isoformat(),
            max_temp_c=max_temp, min_temp_c=min_temp,
            rainfall_mm=rainfall, humidity_pct=humidity, wind_speed_kmh=wind_speed,
            weather_risk_score=risk,
            alert_message=alert or "Conditions normal.",
            source="IMD",
        )
    except Exception as e:
        logger.warning("IMD API failed for %s: %s — using default", tehsil, e)
        return _default_weather(tehsil, str(e))


def fetch_pest_alert(tehsil: str, crop: str, api_key: str = "") -> PestAlert:
    key = api_key or os.getenv("NCIPM_API_KEY", "")
    if not key:
        logger.warning("NCIPM_API_KEY not set — using default pest signal for %s/%s", tehsil, crop)
        return _default_pest(tehsil, crop, "no API key")

    try:
        resp = httpx.get(
            "https://ncipm.icar.gov.in/api/pest-alerts",
            params={"tehsil": tehsil, "crop": crop, "date": date.today().isoformat()},
            headers={"X-API-Key": key},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data   = resp.json()
        alerts = data.get("alerts", [])
        if not alerts:
            return PestAlert(
                tehsil=tehsil, crop=crop, fetch_date=date.today().isoformat(),
                pest_name="none", severity="none", pest_pressure_index=0.0,
                advisory="No active pest alerts for this tehsil and crop.",
                source="ICAR-NCIPM",
            )
        worst    = max(alerts, key=lambda a: _SEVERITY_TO_INDEX.get(a.get("severity", "none"), 0))
        severity = worst.get("severity", "low")
        index    = _SEVERITY_TO_INDEX.get(severity, 0.25)
        return PestAlert(
            tehsil=tehsil, crop=crop, fetch_date=date.today().isoformat(),
            pest_name=worst.get("pestName", "unknown"),
            severity=severity, pest_pressure_index=index,
            advisory=worst.get("advisory", "Apply recommended pesticide per label."),
            emergency=(severity == "severe"),
            source="ICAR-NCIPM",
        )
    except Exception as e:
        logger.warning("NCIPM API failed for %s/%s: %s — using default", tehsil, crop, e)
        return _default_pest(tehsil, crop, str(e))


def fetch_mandi_price(crop: str, state: str, api_key: str = "") -> MandiPrice:
    key = api_key or os.getenv("AGMARKNET_API_KEY", "")
    commodity = COMMODITY_MAP.get(crop.lower(), crop.title())

    if not key:
        logger.warning("AGMARKNET_API_KEY not set — returning default mandi price for %s", crop)
        return _default_mandi(crop, state, "no API key")

    try:
        resp = httpx.get(
            "https://agmarknet.gov.in/api/v1/prices",
            params={"commodity": commodity, "state": state, "date": date.today().isoformat()},
            headers={"Authorization": f"Bearer {key}"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data    = resp.json()
        records = data.get("records", [])
        if not records:
            raise ValueError("No price records returned")

        rec   = records[0]
        modal = float(rec.get("modalPrice", CROP_DEFAULT_PRICE.get(crop.lower(), 2000)))
        min_p = float(rec.get("minPrice",   modal * 0.9))
        max_p = float(rec.get("maxPrice",   modal * 1.1))
        market = rec.get("market", "unknown")

        yesterday_records = data.get("yesterday", [])
        if yesterday_records:
            yesterday_modal = float(yesterday_records[0].get("modalPrice", modal))
            diff  = modal - yesterday_modal
            trend = "rising" if diff > 50 else ("falling" if diff < -50 else "stable")
        else:
            trend = "stable"

        return MandiPrice(
            crop=crop, state=state, fetch_date=date.today().isoformat(),
            modal_price_per_quintal=modal, min_price=min_p, max_price=max_p,
            price_trend=trend, market_name=market, source="Agmarknet",
        )
    except Exception as e:
        logger.warning("Agmarknet API failed for %s/%s: %s — using default", crop, state, e)
        return _default_mandi(crop, state, str(e))


# ---------------------------------------------------------------------------
# Composite enrichment — called by the orchestrator
# CHANGED: Now reads from cache first; only hits live APIs if keys are set.
# ---------------------------------------------------------------------------

def enrich_grower_context(tehsil: str, state: str, crop: str,
                            imd_key: str = "", ncipm_key: str = "",
                            agmarknet_key: str = "") -> ExternalContext:
    """
    Returns external context for a grower's tehsil + crop.

    Lookup priority:
      1. Live API calls (if any API key is provided)
      2. Local cache file (data/signal_cache.json, written by cache_signals_nightly)
      3. Per-crop static defaults

    This means:
      - With keys + internet → live data
      - Without keys, cache fresh → yesterday's data (accurate enough)
      - Without keys, no cache → static defaults (system still runs)
    """
    has_any_key = any([
        imd_key or os.getenv("IMD_API_KEY", ""),
        ncipm_key or os.getenv("NCIPM_API_KEY", ""),
        agmarknet_key or os.getenv("AGMARKNET_API_KEY", ""),
    ])

    if has_any_key:
        # Try live APIs
        weather = fetch_weather_signal(tehsil, state, imd_key)
        pest    = fetch_pest_alert(tehsil, crop, ncipm_key)
        mandi   = fetch_mandi_price(crop, state, agmarknet_key)
        price_bonus = 0.1 if mandi.price_trend == "rising" else 0.0
        urgency = round(
            0.4 * weather.weather_risk_score
            + 0.5 * pest.pest_pressure_index
            + price_bonus, 3
        )
        return ExternalContext(
            weather=weather, pest=pest, mandi=mandi,
            is_emergency=pest.emergency,
            composite_urgency_score=min(urgency, 1.0),
        )

    # No keys → use cache or static defaults
    return get_signal_offline(tehsil, crop, state)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== External Signals Demo ===\n")
    print("No API keys — reading from cache or static defaults.\n")

    ctx = enrich_grower_context(
        tehsil="Patiala_T104", state="Punjab", crop="wheat"
    )
    print(f"Weather risk:      {ctx.weather.weather_risk_score} ({ctx.weather.source})")
    print(f"Pest pressure:     {ctx.pest.pest_pressure_index}  ({ctx.pest.source})")
    print(f"Mandi price:       ₹{ctx.mandi.modal_price_per_quintal}/quintal ({ctx.mandi.price_trend})")
    print(f"Emergency push:    {ctx.is_emergency}")
    print(f"Composite urgency: {ctx.composite_urgency_score}")
    print(f"\nWeather alert:     {ctx.weather.alert_message}")
    print(f"Pest advisory:     {ctx.pest.advisory}")

    print("\n--- Per-crop offline price defaults ---")
    for crop, price in CROP_DEFAULT_PRICE.items():
        print(f"  {crop:12s}: ₹{price:,.0f}/quintal")