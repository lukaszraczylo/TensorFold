#!/usr/bin/env python3
"""analyze_nsys.py <report.sqlite> : kernel share, idle gaps and launch counts from an nsys export."""
import collections
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
tabs = {r[0] for r in cur.execute("select name from sqlite_master where type='table'")}
strs = dict(cur.execute("select id, value from StringIds"))
cols = [r[1] for r in cur.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
nm = "demangledName" if "demangledName" in cols else "shortName"
rows = cur.execute(
    f"select start, end, {nm}, streamId, deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start"
).fetchall()
if not rows:
    sys.exit("no kernels")
t0, t1 = rows[0][0], rows[-1][1]
span = (t1 - t0) / 1e6
busy = sum(e - s for s, e, *_ in rows) / 1e6
print(f"kernels {len(rows)}  span {span:.1f} ms  sum(kernel) {busy:.1f} ms  (streams {len({r[3] for r in rows})})")

agg = collections.defaultdict(lambda: [0, 0])
for s, e, n, *_ in rows:
    k = strs.get(n, str(n)).split("(")[0][:90]
    agg[k][0] += 1
    agg[k][1] += e - s
tot = sum(v[1] for v in agg.values())
print("\nTOP KERNELS by total time (share of sum):")
for k, (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:28]:
    print(f"{100 * d / tot:5.1f}%  {d / 1e6:8.2f} ms  n={c:6d}  avg {d / c / 1e3:8.1f} us  {k}")

by = collections.defaultdict(list)
for s, e, n, st_, dv in rows:
    by[st_].append((s, e))
main = max(by, key=lambda k: sum(e - s for s, e in by[k]))
seq = sorted(by[main])
gaps = [seq[i + 1][0] - seq[i][1] for i in range(len(seq) - 1)]
pos = [g for g in gaps if g > 0]
print(f"\nBUSIEST STREAM {main}: {len(seq)} kernels, gaps: n={len(pos)} total {sum(pos) / 1e6:.2f} ms "
      f"({100 * sum(pos) / (seq[-1][1] - seq[0][0]):.1f}% of its span), overlapped {sum(1 for g in gaps if g < 0)}")
for lo, hi in ((0, 2e3), (2e3, 5e3), (5e3, 20e3), (20e3, 100e3), (100e3, 1e9)):
    g = [x for x in pos if lo <= x < hi]
    print(f"  gap {lo / 1e3:>5.0f}-{hi / 1e3:<6.0f} us: n={len(g):6d} total {sum(g) / 1e6:7.2f} ms")
print("  largest gaps (us):", sorted((round(g / 1e3, 1) for g in pos), reverse=True)[:6])

if "NVTX_EVENTS" in tabs:
    a = collections.defaultdict(lambda: [0, 0])
    for t, d in cur.execute("select text, end-start from NVTX_EVENTS where end is not null"):
        a[t][0] += 1
        a[t][1] += d
    print("\nNVTX ranges:")
    for t, (c, d) in sorted(a.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {d / 1e6:8.2f} ms n={c:5d} {t}")
