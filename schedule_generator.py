import json, math, pandas as pd
from datetime import datetime, timedelta, time

# -----------------------------------------------------------
#  PARAMETERS – adjust once
# -----------------------------------------------------------
AVG_SPEED_KMH = 22
DWELL_SEC     = 18
SERVICE_START = time(6, 0)
SERVICE_END   = time(22, 30)
HEADWAY_MIN   = 10                 # quick test: every 10 min
SERVICE_DATE  = datetime(2025, 1, 1)   # any dummy day works

# after you set SERVICE_START/END
HEADWAY_BY_HOUR = { 6:10, 7:6, 8:6, 9:8, 10:12, 11:8, 12:8,
                    13:10, 14:12, 15:12, 16:8, 17:8, 18:12,
                    19:12, 20:15, 21:15, 22:15 }

def headway_for_dep(dt):
    return HEADWAY_BY_HOUR.get(dt.hour, 20)   # 20 min fallback


# -----------------------------------------------------------
#  Small helper: haversine kilometres
# -----------------------------------------------------------
def hav_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dphi  = math.radians(lat2 - lat1)
    dlamb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlamb/2)**2
    return 2*R*math.asin(math.sqrt(a))

# -----------------------------------------------------------
#  1. Load one route out of routes_output.json
# -----------------------------------------------------------
with open("routes_output.json", encoding="utf-8") as f:
    route = json.load(f)[0]     # use the very first route for now

stops = sorted(route["stops"], key=lambda s: s["seq"])

# build a *minutes per segment* list
seg_min = []
for a, b in zip(stops[:-1], stops[1:]):
    lat1, lon1 = a["geometry"]["coordinates"][1], a["geometry"]["coordinates"][0]
    lat2, lon2 = b["geometry"]["coordinates"][1], b["geometry"]["coordinates"][0]
    seg_min.append(hav_km(lat1, lon1, lat2, lon2) / AVG_SPEED_KMH * 60)

# -----------------------------------------------------------
#  2. Produce trips every HEADWAY_MIN minutes
# -----------------------------------------------------------
records = []
trip_num = 1
dep_clock = datetime.combine(SERVICE_DATE, SERVICE_START)

while dep_clock.time() <= SERVICE_END:
    trip_id = f"T{trip_num:05d}"
    cursor  = dep_clock                       # <-- here’s the variable!
    
    for idx, st in enumerate(stops):
        hw = headway_for_dep(dep_clock)
        arr  = cursor
        dep  = arr + timedelta(seconds=DWELL_SEC)
        records.append({
            "trip_id": trip_id,
            "arrival_time":  arr.time().strftime("%H:%M:%S"),
            "departure_time":dep.time().strftime("%H:%M:%S"),
            "stop_id": f"{route['name']}_{st['seq']}",
            "stop_sequence": idx+1
        })
        if idx < len(seg_min):               # move to next stop
            cursor = dep + timedelta(minutes=seg_min[idx])

    # next vehicle in the schedule
    dep_clock += timedelta(minutes=hw)
    trip_num += 1

# -----------------------------------------------------------
#  3. Save tidy CSV (unchanged)
# -----------------------------------------------------------
df = pd.DataFrame(records)
df.to_csv("debug_schedule.csv", index=False)
print("✓ Wrote debug_schedule.csv with", len(df), "rows")

# -----------------------------------------------------------
#  4. Build “human” timetable text per stop
#     Format:  HH - mm mm mm …
# -----------------------------------------------------------
df["hour"]   = df["departure_time"].str.slice(0, 2).astype(int)
df["minute"] = df["departure_time"].str.slice(3, 5).astype(int)

for stop_id in df["stop_id"].unique():
    friendly_lines = []
    friendly_lines.append(f"# Timetable for {stop_id}\n")
    sub = df[df["stop_id"] == stop_id]

    # loop 05 → 23 so empty hours are skipped automatically
    for hour in range(SERVICE_START.hour - 1, SERVICE_END.hour + 2):
        mins = sub[sub["hour"] == hour]["minute"]
        if not mins.empty:
            minute_str = " ".join(f"{m:02d}" for m in mins)
            friendly_lines.append(f"{hour:2d} - {minute_str}")

    # blank line at the end
    friendly_lines.append("")
    fname = f"timetable_{stop_id}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(friendly_lines))
    print(f"✓ Wrote {fname}")
