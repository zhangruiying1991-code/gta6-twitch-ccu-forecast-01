"""GTA VI Twitch CCU forecaster.

Idea, in plain words: hour 1 alone tells us the scale, not the shape.
The shape of a big console launch day is very similar across games:
a dip in the deep night (hours 3-4 UTC), a rise as the US unlocks
(hours 6 and 9), a peak in the US daytime (hours 19-21), then a fall.

So the program reads the comparables shipped in the case file,
keeps only the launches that look like GTA VI -- console-only,
midnight rolling unlock, hour 1 already well above the game's
pre-release level -- and takes, hour by hour, the median log-ratio
to hour 1 across those launches, with the spread of those ratios
as the quantile spread.  Nothing is fetched, nothing is random,
standard library only.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]


def parse_time(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def median_pre(hourly, day_start):
    """Median hourly viewers in the 7 days before day_start."""
    vals = []
    for h in hourly or []:
        try:
            t = parse_time(h["from"])
            v = h.get("viewers")
        except Exception:
            continue
        if v and v > 0 and (day_start.timestamp() - 7 * 86400) <= t.timestamp() < day_start.timestamp():
            vals.append(v)
    return statistics.median(vals) if vals else None


def day_curve(comp):
    """Hour 0..23 viewers on the comparable's release day, or {} if sparse."""
    try:
        ds = parse_time(comp["day_start"])
    except Exception:
        return {}
    hrs = {}
    for h in comp.get("hourly") or []:
        try:
            t = parse_time(h["from"])
            v = h.get("viewers")
        except Exception:
            continue
        d = int((t - ds).total_seconds() // 3600)
        if 0 <= d < 24 and v and v > 0:
            hrs[d] = v
    return hrs


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    # Target's own pre-release level, from its history_hourly.
    try:
        day_start = parse_time(case["day_start"])
    except Exception:
        day_start = None
    target_pre = None
    if day_start is not None:
        target_pre = median_pre(case.get("history_hourly") or [], day_start)

    # Build the pool of comparable launch-day curves.
    pool = []  # (name, hours dict, pre_median)
    for comp in case.get("comparables") or []:
        hrs = day_curve(comp)
        if 0 not in hrs or len(hrs) < 20:
            continue
        try:
            cds = parse_time(comp["day_start"])
        except Exception:
            continue
        pre = median_pre(comp.get("hourly") or [], cds)
        pool.append((comp.get("name", ""), hrs, pre, comp.get("platforms", "")))

    def is_console_like(item):
        _n, hrs, pre, platforms = item
        if "PC" in (platforms or ""):
            return False
        if pre and hrs[0] / pre <= 3:
            return False
        if max(hrs.values()) / hrs[0] > 6:
            return False
        if hrs[0] < 1000:
            return False
        return True

    filtered = [it for it in pool if is_console_like(it)]
    # GTA VI's hour 1 should be a real launch hour, not a pre-launch
    # trickle.  If it is not, fall back to the broad pool with a wide spread.
    target_is_launch = True
    if target_pre and p1 / target_pre < 3:
        target_is_launch = False

    if target_is_launch and len(filtered) >= 5:
        use = filtered
        sd_floor = 0.25
    else:
        # Broad fallback: any launch with a full day, median shape,
        # but a much wider spread because the shape is far less certain.
        use = pool
        sd_floor = 0.50

    hours_out = []
    mus = {}
    sds = {}
    for h in range(1, 24):
        logs = []
        for _n, hrs, _pre, _plat in use:
            if h in hrs and 0 in hrs:
                logs.append(math.log(hrs[h] / hrs[0]))
        if len(logs) < 3:
            mus[h] = 0.0
            sds[h] = max(0.15 * math.sqrt(h), sd_floor)
        else:
            mus[h] = statistics.median(logs)
            sd = statistics.pstdev(logs) if len(logs) > 1 else sd_floor
            sds[h] = min(max(sd, sd_floor), 0.65)

    # Light 3-hour smoothing of the median curve: keeps the night dip
    # and the daytime peak, removes single-hour noise.
    smoothed = {}
    for h in range(1, 24):
        vals = []
        wts = []
        for dh, w in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            if h + dh in mus:
                vals.append(mus[h + dh] * w)
                wts.append(w)
        smoothed[h] = sum(vals) / sum(wts) if wts else mus[h]

    for row in case["forecast"]:
        h = row["hour"]
        mu = smoothed.get(h, 0.0)
        sd = sds.get(h, sd_floor)
        qs = []
        prev = 0.0
        for z in Z:
            q = p1 * math.exp(mu + z * sd)
            q = round2(max(1.0, min(q, 1e9)))
            q = max(q, prev)
            qs.append(q)
            prev = q
        hours_out.append({"hour": h, "q": qs})
    return {"hours": hours_out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
