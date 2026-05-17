"""Generate synthetic sign-in events.

Usage:
    uv run python scripts/gen_signin_events.py <input_path> <n_ingest> <n_score> <n_score_anomalies>
"""
import json
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

SEED = 42
random.seed(SEED)

PERSONAS = [
    {"id": "user_001", "department": "IT",                       "country": "DE", "desktop": ("Windows", "Chrome"),  "mobile": ("iOS",     "Safari")},
    {"id": "user_002", "department": "IT",                       "country": "DE", "desktop": ("Mac",     "Firefox"), "mobile": ("Android", "Chrome")},
    {"id": "user_003", "department": "IT",                       "country": "DE", "desktop": ("Linux",   "Chrome"),  "mobile": ("iOS",     "Firefox")},
    {"id": "user_004", "department": "IT",                       "country": "DE", "desktop": ("Windows", "Edge"),    "mobile": ("Android", "Firefox")},
    {"id": "user_005", "department": "IT",                       "country": "DE", "desktop": ("Linux",   "Firefox"), "mobile": ("iOS",     "Chrome")},
    {"id": "user_006", "department": "IT",                       "country": "ES", "desktop": ("Mac",     "Chrome"),  "mobile": ("Android", "Chrome")},
    {"id": "user_007", "department": "IT",                       "country": "US", "desktop": ("Windows", "Firefox"), "mobile": ("iOS",     "Safari")},
    {"id": "user_008", "department": "Sales",                    "country": "DE", "desktop": ("Mac",     "Safari"),  "mobile": ("iOS",     "Safari")},
    {"id": "user_009", "department": "Sales",                    "country": "DE", "desktop": ("Windows", "Chrome"),  "mobile": ("Android", "Chrome")},
    {"id": "user_010", "department": "Sales",                    "country": "DE", "desktop": ("Linux",   "Firefox"), "mobile": ("iOS",     "Firefox")},
    {"id": "user_011", "department": "Sales",                    "country": "ES", "desktop": ("Windows", "Edge"),    "mobile": ("Android", "Firefox")},
    {"id": "user_012", "department": "Sales",                    "country": "ES", "desktop": ("Mac",     "Chrome"),  "mobile": ("iOS",     "Chrome")},
    {"id": "user_013", "department": "Sales",                    "country": "ES", "desktop": ("Linux",   "Chrome"),  "mobile": ("Android", "Chrome")},
    {"id": "user_014", "department": "Sales",                    "country": "US", "desktop": ("Windows", "Firefox"), "mobile": ("iOS",     "Safari")},
    {"id": "user_015", "department": "Sales",                    "country": "CN", "desktop": ("Mac",     "Firefox"), "mobile": ("Android", "Firefox")},
    {"id": "user_016", "department": "Aftersales",               "country": "US", "desktop": ("Windows", "Chrome"),  "mobile": ("iOS",     "Safari")},
    {"id": "user_017", "department": "Aftersales",               "country": "US", "desktop": ("Mac",     "Safari"),  "mobile": ("Android", "Chrome")},
    {"id": "user_018", "department": "Aftersales",               "country": "US", "desktop": ("Linux",   "Chrome"),  "mobile": ("iOS",     "Firefox")},
    {"id": "user_019", "department": "Aftersales",               "country": "US", "desktop": ("Windows", "Edge"),    "mobile": ("Android", "Firefox")},
    {"id": "user_020", "department": "Aftersales",               "country": "US", "desktop": ("Mac",     "Firefox"), "mobile": ("iOS",     "Chrome")},
    {"id": "user_021", "department": "Aftersales",               "country": "DE", "desktop": ("Windows", "Firefox"), "mobile": ("Android", "Chrome")},
    {"id": "user_022", "department": "Aftersales",               "country": "ES", "desktop": ("Linux",   "Firefox"), "mobile": ("iOS",     "Safari")},
    {"id": "user_023", "department": "Research-and-Development", "country": "DE", "desktop": ("Windows", "Chrome"),  "mobile": ("iOS",     "Safari")},
    {"id": "user_024", "department": "Research-and-Development", "country": "DE", "desktop": ("Mac",     "Chrome"),  "mobile": ("Android", "Chrome")},
    {"id": "user_025", "department": "Research-and-Development", "country": "DE", "desktop": ("Linux",   "Firefox"), "mobile": ("iOS",     "Firefox")},
    {"id": "user_026", "department": "Research-and-Development", "country": "DE", "desktop": ("Windows", "Edge"),    "mobile": ("Android", "Firefox")},
    {"id": "user_027", "department": "Research-and-Development", "country": "DE", "desktop": ("Mac",     "Firefox"), "mobile": ("iOS",     "Chrome")},
    {"id": "user_028", "department": "Research-and-Development", "country": "ES", "desktop": ("Linux",   "Chrome"),  "mobile": ("Android", "Chrome")},
    {"id": "user_029", "department": "Research-and-Development", "country": "US", "desktop": ("Windows", "Firefox"), "mobile": ("iOS",     "Safari")},
    {"id": "user_030", "department": "Product-A",                "country": "DE", "desktop": ("Mac",     "Safari"),  "mobile": ("Android", "Chrome")},
    {"id": "user_031", "department": "Product-A",                "country": "DE", "desktop": ("Windows", "Chrome"),  "mobile": ("iOS",     "Firefox")},
    {"id": "user_032", "department": "Product-A",                "country": "DE", "desktop": ("Linux",   "Chrome"),  "mobile": ("Android", "Firefox")},
    {"id": "user_033", "department": "Product-A",                "country": "ES", "desktop": ("Windows", "Edge"),    "mobile": ("iOS",     "Chrome")},
    {"id": "user_034", "department": "Product-A",                "country": "ES", "desktop": ("Mac",     "Chrome"),  "mobile": ("Android", "Chrome")},
    {"id": "user_035", "department": "Product-A",                "country": "ES", "desktop": ("Linux",   "Firefox"), "mobile": ("iOS",     "Safari")},
    {"id": "user_036", "department": "Product-A",                "country": "US", "desktop": ("Windows", "Firefox"), "mobile": ("Android", "Firefox")},
    {"id": "user_037", "department": "Product-B",                "country": "CN", "desktop": ("Windows", "Chrome"),  "mobile": ("Android", "Chrome")},
    {"id": "user_038", "department": "Product-B",                "country": "CN", "desktop": ("Mac",     "Chrome"),  "mobile": ("iOS",     "Safari")},
    {"id": "user_039", "department": "Product-B",                "country": "CN", "desktop": ("Linux",   "Firefox"), "mobile": ("Android", "Firefox")},
    {"id": "user_040", "department": "Product-B",                "country": "CN", "desktop": ("Windows", "Edge"),    "mobile": ("iOS",     "Chrome")},
    {"id": "user_041", "department": "Product-B",                "country": "CN", "desktop": ("Mac",     "Firefox"), "mobile": ("Android", "Chrome")},
    {"id": "user_042", "department": "Product-B",                "country": "DE", "desktop": ("Linux",   "Chrome"),  "mobile": ("iOS",     "Firefox")},
    {"id": "user_043", "department": "Product-B",                "country": "US", "desktop": ("Windows", "Firefox"), "mobile": ("Android", "Firefox")},
    {"id": "user_044", "department": "Product-C",                "country": "US", "desktop": ("Mac",     "Safari"),  "mobile": ("iOS",     "Safari")},
    {"id": "user_045", "department": "Product-C",                "country": "US", "desktop": ("Windows", "Chrome"),  "mobile": ("Android", "Chrome")},
    {"id": "user_046", "department": "Product-C",                "country": "US", "desktop": ("Linux",   "Chrome"),  "mobile": ("iOS",     "Firefox")},
    {"id": "user_047", "department": "Product-C",                "country": "US", "desktop": ("Windows", "Edge"),    "mobile": ("Android", "Firefox")},
    {"id": "user_048", "department": "Product-C",                "country": "US", "desktop": ("Mac",     "Firefox"), "mobile": ("iOS",     "Chrome")},
    {"id": "user_049", "department": "Product-C",                "country": "DE", "desktop": ("Windows", "Firefox"), "mobile": ("Android", "Chrome")},
    {"id": "user_050", "department": "Product-C",                "country": "CN", "desktop": ("Mac",     "Chrome"),  "mobile": ("iOS",     "Safari")},
]

LATENCY_RANGE = {
    "DE": (20, 50),
    "ES": (30, 60),
    "US": (100, 140),
    "CN": (150, 200),
}

# UTC offset (hours) per country — simplified fixed offsets
TZ_OFFSET = {"DE": 1, "ES": 1, "US": -5, "CN": 8}

# Precompute 2025 weekday / weekend date pools once
_ALL_2025 = [date(2025, 1, 1) + timedelta(days=i) for i in range(365)]
_WEEKDAYS = [d for d in _ALL_2025 if d.weekday() < 5]   # Mon–Fri
_WEEKENDS = [d for d in _ALL_2025 if d.weekday() >= 5]  # Sat–Sun


def random_timestamp(department: str, country: str) -> str:
    """Return a UTC ISO-8601 timestamp that respects business-hour rules.

    - All departments: weekdays 08:00–17:00 local time.
    - R&D: 5 % chance the event falls on a weekend (still 08:00–17:00 local).
    - Sales: 10 % chance the event falls in the evening 17:00–22:00 local
      (weekdays only; takes precedence over the standard window).
    """
    offset = TZ_OFFSET[country]

    is_weekend_event = (department == "Research-and-Development"
                        and random.random() < 0.05)
    is_evening_event = (not is_weekend_event
                        and department == "Sales"
                        and random.random() < 0.10)

    day = random.choice(_WEEKENDS if is_weekend_event else _WEEKDAYS)

    if is_evening_event:
        hour = random.randint(17, 21)   # 17:00–21:59 local
    else:
        hour = random.randint(8, 16)    # 08:00–16:59 local

    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    local_dt = datetime(day.year, day.month, day.day, hour, minute, second)
    utc_dt = local_dt - timedelta(hours=offset)

    # Edge case: US evening on Dec 31 can spill into Jan 1 2026 — retry once.
    if utc_dt.year != 2025:
        return random_timestamp(department, country)

    return utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


_ALL_OS_BROWSER: list[tuple[str, str]] = [
    ("Windows", "Chrome"), ("Windows", "Edge"),   ("Windows", "Firefox"),
    ("Linux",   "Chrome"), ("Linux",   "Firefox"),
    ("Mac",     "Chrome"), ("Mac",     "Firefox"), ("Mac",    "Safari"),
    ("iOS",     "Chrome"), ("iOS",     "Firefox"), ("iOS",    "Safari"),
    ("Android", "Chrome"), ("Android", "Firefox"),
]


def make_event(persona: dict) -> dict:
    lo, hi = LATENCY_RANGE[persona["country"]]
    use_mobile = random.random() < 0.25  # 25% mobile, 75% desktop
    os_name, browser = persona["mobile"] if use_mobile else persona["desktop"]
    return {
        "timestamp": random_timestamp(persona["department"], persona["country"]),
        "user": {"id": persona["id"], "department": persona["department"]},
        "ip":   {"country_iso": persona["country"]},
        "client": {"os": os_name, "browser": browser},
        "latency": random.randint(lo, hi),
    }


def make_anomaly_event(persona: dict) -> dict:
    """Normal event for this persona with exactly one anomalous property.

    Picks randomly from four anomaly types:
      country    — logs in from a different country; latency matches that country.
      os_browser — uses an OS/Browser combo outside the persona's desktop & mobile pair.
      time       — logs in at night (22:00–05:59 local), when the user normally does not work.
      weekend    — logs in on a Saturday or Sunday during normal hours.
                   Not available for Research-and-Development (weekend work is normal for them).

    user.id and department are never changed.
    """
    available = ["country", "os_browser", "time"]
    if persona["department"] != "Research-and-Development":
        available.append("weekend")
    anomaly_type = random.choice(available)

    use_mobile = random.random() < 0.25
    os_name, browser = persona["mobile"] if use_mobile else persona["desktop"]

    if anomaly_type == "country":
        other_countries = [c for c in LATENCY_RANGE if c != persona["country"]]
        country = random.choice(other_countries)
        lo, hi  = LATENCY_RANGE[country]
        return {
            "timestamp":   random_timestamp(persona["department"], country),
            "user":        {"id": persona["id"], "department": persona["department"]},
            "ip":          {"country_iso": country},
            "client":      {"os": os_name, "browser": browser},
            "latency":     random.randint(lo, hi),
            "description": "ANOMALY:country",
        }

    if anomaly_type == "os_browser":
        normal = {persona["desktop"], persona["mobile"]}
        unusual_combos = [c for c in _ALL_OS_BROWSER if c not in normal]
        os_name, browser = random.choice(unusual_combos)
        lo, hi = LATENCY_RANGE[persona["country"]]
        return {
            "timestamp":   random_timestamp(persona["department"], persona["country"]),
            "user":        {"id": persona["id"], "department": persona["department"]},
            "ip":          {"country_iso": persona["country"]},
            "client":      {"os": os_name, "browser": browser},
            "latency":     random.randint(lo, hi),
            "description": "ANOMALY:os_browser",
        }

    if anomaly_type == "weekend":
        offset = TZ_OFFSET[persona["country"]]
        while True:
            day      = random.choice(_WEEKENDS)
            hour     = random.randint(8, 16)
            minute   = random.randint(0, 59)
            second   = random.randint(0, 59)
            local_dt = datetime(day.year, day.month, day.day, hour, minute, second)
            utc_dt   = local_dt - timedelta(hours=offset)
            if utc_dt.year == 2025:
                break
        lo, hi = LATENCY_RANGE[persona["country"]]
        return {
            "timestamp":   utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "user":        {"id": persona["id"], "department": persona["department"]},
            "ip":          {"country_iso": persona["country"]},
            "client":      {"os": os_name, "browser": browser},
            "latency":     random.randint(lo, hi),
            "description": "ANOMALY:weekend",
        }

    # time anomaly — night shift (22:00–05:59 local)
    offset = TZ_OFFSET[persona["country"]]
    while True:
        day    = random.choice(_WEEKDAYS)
        hour   = random.choice(list(range(22, 24)) + list(range(0, 6)))
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        local_dt = datetime(day.year, day.month, day.day, hour, minute, second)
        utc_dt   = local_dt - timedelta(hours=offset)
        if utc_dt.year == 2025:
            break
    lo, hi = LATENCY_RANGE[persona["country"]]
    return {
        "timestamp":   utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user":        {"id": persona["id"], "department": persona["department"]},
        "ip":          {"country_iso": persona["country"]},
        "client":      {"os": os_name, "browser": browser},
        "latency":     random.randint(lo, hi),
        "description": "ANOMALY:time",
    }


def main() -> None:
    if len(sys.argv) < 5:
        print("Usage: uv run python scripts/gen_signin_events.py <input_path> <n_ingest> <n_score> <n_score_anomalies>", file=sys.stderr)
        raise SystemExit("Error: input_path, n_ingest, n_score, and n_score_anomalies are required.")

    repo_root  = Path(__file__).parent.parent
    input_path = Path(sys.argv[1])
    if not input_path.is_absolute():
        input_path = repo_root / input_path

    try:
        n_ingest           = int(sys.argv[2])
        n_score            = int(sys.argv[3])
        n_score_anomalies  = int(sys.argv[4])
    except ValueError:
        raise SystemExit("Error: n_ingest, n_score, and n_score_anomalies must be integers.")

    if input_path.exists():
        with open(input_path) as f:
            data = json.load(f)
    else:
        data = {"ingest": [], "scores": []}

    data["ingest"].extend(make_event(random.choice(PERSONAS)) for _ in range(n_ingest))
    data["ingest"].sort(key=lambda e: e["timestamp"])

    data["scores"].extend(
        {**make_event(random.choice(PERSONAS)), "description": "NORMAL"}
        for _ in range(n_score)
    )
    data["scores"].extend(
        make_anomaly_event(random.choice(PERSONAS))
        for _ in range(n_score_anomalies)
    )
    data["scores"].sort(key=lambda e: e["timestamp"])

    with open(input_path, "w") as f:
        json.dump(data, f, indent=2)

    n_anomalies = sum(1 for e in data["scores"] if e.get("description", "").startswith("ANOMALY"))
    n_normal    = sum(1 for e in data["scores"] if e.get("description") == "NORMAL")
    print(f"Done — {len(data['ingest'])} ingest, {n_normal} normal scores, {n_anomalies} anomaly scores.")


if __name__ == "__main__":
    main()
