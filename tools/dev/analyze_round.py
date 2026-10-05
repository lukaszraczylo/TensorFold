#!/usr/bin/env python3
"""analyze_round.py <report.sqlite> [round-index] [list-kernels]: one decode round's critical path from an nsys export.

Finds the busiest stream (the engine's), splits it into rounds at idle gaps over 400 us, and for the chosen round
(default the middle one) prints wall time, kernel time by name, idle time between kernels and, with ``list-kernels``,
the kernels in order with their offset, duration and the gap to the next one.
"""

import collections
import sqlite3
import sys

GAP_ROUND_US = 400.0

db = sqlite3.connect(sys.argv[1])
cur = db.cursor()
strs = dict(cur.execute("select id, value from StringIds"))
cols = [r[1] for r in cur.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
nm = "demangledName" if "demangledName" in cols else "shortName"
rows = cur.execute(
    f"select start, end, {nm}, streamId, deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by start"
).fetchall()
per = collections.Counter((r[4], r[3]) for r in rows)
(dev, stream), _ = per.most_common(1)[0]
main = [(s, e, strs.get(n, str(n)).split("(")[0][:60]) for s, e, n, st, d in rows if (d, st) == (dev, stream)]

rounds, cur_round = [], [main[0]]
for k in main[1:]:
    if (k[0] - cur_round[-1][1]) / 1e3 > GAP_ROUND_US:
        rounds.append(cur_round)
        cur_round = []
    cur_round.append(k)
rounds.append(cur_round)
decode = [r for r in rounds if sum("dec_kernel" in k[2] for k in r) >= 40]
print(f"device {dev} stream {stream}: {len(main)} kernels, {len(rounds)} segments, {len(decode)} decode rounds")
idx = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else len(decode) // 2
r = decode[idx]
wall = (r[-1][1] - r[0][0]) / 1e6
busy = sum(e - s for s, e, _ in r) / 1e6
gaps = [(r[i + 1][0] - r[i][1]) / 1e3 for i in range(len(r) - 1)]   # us
print(f"round {idx}: {len(r)} kernels, wall {wall:.2f} ms, kernel time {busy:.2f} ms, gaps {sum(g for g in gaps if g > 0) / 1e3:.2f} ms "
      f"(over 20 us: {sum(g for g in gaps if g > 20) / 1e3:.2f} ms in {sum(g > 20 for g in gaps)})")
agg = collections.defaultdict(lambda: [0, 0.0])
for s, e, n in r:
    agg[n][0] += 1
    agg[n][1] += (e - s) / 1e6
print("\nkernel                                                        n   ms    avg us")
for n, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:28]:
    print(f"{n:58s} {c:5d} {t:6.2f} {1e3 * t / c:8.1f}")
big = sorted(((g, i) for i, g in enumerate(gaps) if g > 20), reverse=True)[:8]
print("\nlargest idle gaps (us, after kernel):", [(round(g, 1), r[i][2][:28]) for g, i in big])
if "list-kernels" in sys.argv:
    t0 = r[0][0]
    print("\n offset_us   dur_us   gap_us  kernel")
    for i, (s, e, n) in enumerate(r[:260]):
        gap = (r[i + 1][0] - e) / 1e3 if i + 1 < len(r) else 0.0   # us
        print(f"{(s - t0) / 1e3:9.1f} {(e - s) / 1e3:8.1f} {gap:8.1f}  {n}")
