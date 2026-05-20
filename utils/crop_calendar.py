"""
Utility: Crop Calendar & Growth Stage Calculator
Parses grower_crop_calendar JSON to determine current growth stage
and days until the next critical stage.
"""

from datetime import datetime
from typing import Optional


def get_growth_stage(stages: list, today: datetime) -> str:
    """
    Determine the current growth stage based on the crop calendar stages JSON.
    Returns the most recently passed stage, or 'pre-sowing' if none passed yet.

    Args:
        stages: List of {"stage": "tillering", "approx": "2026-01-15"} dicts
        today: Reference datetime

    Returns:
        Current growth stage name (str)
    """
    if not stages:
        return "vegetative"

    passed = []
    for s in stages:
        try:
            stage_date = datetime.strptime(s["approx"], "%Y-%m-%d")
            if stage_date <= today:
                passed.append((stage_date, s["stage"]))
        except (KeyError, ValueError):
            continue

    if not passed:
        return "pre-sowing"

    # Return the most recently passed stage
    passed.sort(key=lambda x: x[0])
    return passed[-1][1]


def days_to_next_stage(stages: list, today: datetime) -> int:
    """
    Calculate days until the next upcoming growth stage.

    Args:
        stages: List of {"stage": "...", "approx": "YYYY-MM-DD"} dicts
        today: Reference datetime

    Returns:
        Number of days to next stage; -1 if no future stages
    """
    if not stages:
        return -1

    future = []
    for s in stages:
        try:
            stage_date = datetime.strptime(s["approx"], "%Y-%m-%d")
            delta = (stage_date - today).days
            if delta > 0:
                future.append(delta)
        except (KeyError, ValueError):
            continue

    return min(future) if future else -1


def get_upcoming_stage(stages: list, today: datetime) -> Optional[dict]:
    """
    Return the next upcoming stage dict (with stage name and date).

    Returns:
        {"stage": "flowering", "approx": "2026-02-20", "days_away": 7} or None
    """
    if not stages:
        return None

    future = []
    for s in stages:
        try:
            stage_date = datetime.strptime(s["approx"], "%Y-%m-%d")
            delta = (stage_date - today).days
            if delta > 0:
                future.append({"stage": s["stage"], "approx": s["approx"], "days_away": delta})
        except (KeyError, ValueError):
            continue

    if not future:
        return None

    return min(future, key=lambda x: x["days_away"])


def is_critical_window(stages: list, today: datetime, window_days: int = 7) -> bool:
    """
    Returns True if the next critical growth stage is within `window_days`.
    Used to trigger high-priority campaign sends.

    Args:
        stages: List of stage dicts from grower_crop_calendar
        today: Reference datetime
        window_days: Number of days to look ahead (default: 7)

    Returns:
        True if a stage is coming within window_days
    """
    days = days_to_next_stage(stages, today)
    return 0 < days <= window_days


def parse_crop_calendar(calendar_json: str) -> dict:
    """
    Parse the grower_crop_calendar JSON string into a structured dict.

    Returns:
        {
          "season": "Rabi_2025-26",
          "crop": "wheat",
          "sowing_start": "2025-11-01",
          "sowing_end": "2025-11-25",
          "harvest_start": "2026-03-20",
          "harvest_end": "2026-04-15",
          "stages": [{"stage": "tillering", "approx": "2026-01-15"}, ...],
          "current_stage": "tillering",
          "days_to_next_stage": 14,
          "upcoming_stage": {...} | None,
          "is_critical_window": False,
        }
    """
    import json

    try:
        cal = json.loads(calendar_json) if isinstance(calendar_json, str) else calendar_json
    except (json.JSONDecodeError, TypeError):
        return {}

    today = datetime.utcnow()
    stages = cal.get("stages", [])

    return {
        "season": cal.get("season", ""),
        "crop": cal.get("crop", "unknown"),
        "sowing_start": cal.get("sowing", {}).get("start", ""),
        "sowing_end": cal.get("sowing", {}).get("end", ""),
        "harvest_start": cal.get("harvest", {}).get("start", ""),
        "harvest_end": cal.get("harvest", {}).get("end", ""),
        "stages": stages,
        "current_stage": get_growth_stage(stages, today),
        "days_to_next_stage": days_to_next_stage(stages, today),
        "upcoming_stage": get_upcoming_stage(stages, today),
        "is_critical_window": is_critical_window(stages, today, window_days=7),
    }


def season_progress(sowing_start_str: str, harvest_start_str: str,
                    today: Optional[datetime] = None) -> float:
    """
    Compute the normalized progress through the crop season (0.0 to 1.0).

    Args:
        sowing_start_str: ISO date string of sowing start
        harvest_start_str: ISO date string of harvest start
        today: Reference datetime (defaults to UTC now)

    Returns:
        Float between 0 and 1
    """
    if today is None:
        today = datetime.utcnow()

    try:
        sowing = datetime.strptime(sowing_start_str, "%Y-%m-%d")
        harvest = datetime.strptime(harvest_start_str, "%Y-%m-%d")
        total = (harvest - sowing).days
        if total <= 0:
            return 0.5
        elapsed = (today - sowing).days
        return float(max(0.0, min(1.0, elapsed / total)))
    except (ValueError, TypeError):
        return 0.5


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    from datetime import datetime

    sample_calendar = json.dumps({
        "season": "Rabi_2025-26",
        "crop": "wheat",
        "sowing": {"start": "2025-11-01", "end": "2025-11-25"},
        "harvest": {"start": "2026-03-20", "end": "2026-04-15"},
        "stages": [
            {"stage": "tillering", "approx": "2026-01-15"},
            {"stage": "flowering", "approx": "2026-02-20"},
        ]
    })

    today = datetime(2026, 2, 14)
    parsed = parse_crop_calendar(sample_calendar)
    print("Parsed calendar:")
    for k, v in parsed.items():
        print(f"  {k}: {v}")

    print(f"\nSeason progress: {season_progress('2025-11-01', '2026-03-20', today):.1%}")
    print(f"Critical window (7 days): {is_critical_window(parsed['stages'], today, window_days=7)}")
