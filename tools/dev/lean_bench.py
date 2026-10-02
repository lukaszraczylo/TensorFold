#!/usr/bin/env python3
"""lean_bench.py <arm> <base-url> <out.json> [want_blocks=4] [max_blocks=12]

Single-stream decode speed through the pinned upstream harness (glm53/bench/tf_bench.py in home-kubernetes):
alternating prose/code runs, 400 tokens, temperature 0, thinking off. One discard pair first.

The embedders serve real traffic, and their load level moves a result by about 3.5%. A block is 5 prose/code pairs.
The embedder load (millicores, busiest pod) is read before and after each block, and the block is tagged quiet (both
readings under QUIET_M), loaded (both at or over LOADED_M) or transition. The run goes on until `want_blocks`
blocks of one regime exist or `max_blocks` have run. The result is the median over the blocks of the regime with
most blocks. Compare a result only with a baseline of the same regime.
"""
import json
import statistics as st
import sys
import time

BENCH = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "bench")
sys.path.insert(0, BENCH)
import drafter_ab as ab  # noqa: E402
import quiet_bench as qb  # noqa: E402
import tf_bench  # noqa: E402

QUIET_M = 100
LOADED_M = 250
arm, base, out = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]
want = int(sys.argv[4]) if len(sys.argv) > 4 else 4
max_blocks = int(sys.argv[5]) if len(sys.argv) > 5 else 12
bd, _ = ab.load_harness(qb.UPSTREAM, False)
bd.BASE = base
sets = ab.prompt_sets(bd)


def regime(e0, e1):
    if e0 is None or e1 is None:
        return "unknown"
    if e0 < QUIET_M and e1 < QUIET_M:
        return "quiet"
    return "loaded" if e0 >= LOADED_M and e1 >= LOADED_M else "transition"


def pair():
    got = {}
    for name in ("prose", "code"):
        r = tf_bench.one_run(bd, sets[name][0], sets[name][1], 400)
        got[name] = None if r["void"] else r["tok_s"]
    return got


pair()  # discard
blocks = []
for b in range(max_blocks):
    e0 = qb.embed_m()
    runs = {"prose": [], "code": []}
    for _ in range(5):
        for k, v in pair().items():
            if v is not None:
                runs[k].append(v)
        time.sleep(0.3)
    e1 = qb.embed_m()
    blk = {"embed": [e0, e1], "regime": regime(e0, e1), "runs": runs,
           "prose": st.median(runs["prose"]), "code": st.median(runs["code"])}
    blocks.append(blk)
    print(f"block {b}: embed {e0}->{e1}m {blk['regime']} prose {blk['prose']:.2f} code {blk['code']:.2f}", flush=True)
    if max(sum(1 for x in blocks if x["regime"] == k) for k in ("quiet", "loaded")) >= want:
        break

counts = {r: sum(1 for x in blocks if x["regime"] == r) for r in ("quiet", "loaded")}
best = max(counts, key=counts.get)
sel = [x for x in blocks if x["regime"] == best]
res = {"arm": arm, "regime": best, "n_blocks": len(sel), "blocks": blocks}
if sel:
    res["prose"] = st.median(x["prose"] for x in sel)
    res["code"] = st.median(x["code"] for x in sel)
    res["pooled"] = (res["prose"] + res["code"]) / 2
json.dump(res, open(out, "w"))
print(f"LEAN {arm} regime={best} blocks={len(sel)}/{len(blocks)} POOLED {res.get('pooled', 0):.3f} "
      f"prose {res.get('prose', 0):.2f} code {res.get('code', 0):.2f}")
