#!/usr/bin/env python3
"""lean_bench.py <arm> <base-url> <out.json> [pairs=20]

Alternate prose/code single-stream runs back to back (400 tokens, temperature 0, thinking off) through the pinned
upstream harness (glm53/bench/tf_bench.py in home-kubernetes). The first pair is a discard. The embedder load
(millicores, busiest pod) is sampled at the start of every block of 5 pairs and at the end, so a result can be
compared against a baseline from the same load regime: quiet is under 100m, loaded is 300m and over.
"""
import json
import statistics as st
import sys
import time

BENCH = "/Users/nvm/Documents/projects/home-cluster/.wt-dev/namespaces/llm/base/glm53/bench"
sys.path.insert(0, BENCH)
import drafter_ab as ab  # noqa: E402
import quiet_bench as qb  # noqa: E402
import tf_bench  # noqa: E402

arm, base, out = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]
pairs = int(sys.argv[4]) if len(sys.argv) > 4 else 20
bd, _ = ab.load_harness(qb.UPSTREAM, False)
bd.BASE = base
sets = ab.prompt_sets(bd)
runs = {"prose": [], "code": []}
embed = []
for i in range(pairs + 1):
    if i % 5 == 0:
        embed.append(qb.embed_m())
    for name in ("prose", "code"):
        r = tf_bench.one_run(bd, sets[name][0], sets[name][1], 400)
        if i == 0 or r["void"]:
            continue
        runs[name].append(r["tok_s"])
    time.sleep(0.5)
embed.append(qb.embed_m())
blocks = []
for b in range(0, pairs, 5):
    p, c = runs["prose"][b:b + 5], runs["code"][b:b + 5]
    if p and c:
        blocks.append((round(st.median(p), 2), round(st.median(c), 2)))
mp, mc = st.median(runs["prose"]), st.median(runs["code"])
res = {"arm": arm, "prose": mp, "code": mc, "pooled": (mp + mc) / 2, "blocks": blocks, "embed_m": embed, "runs": runs}
json.dump(res, open(out, "w"))
regime = "quiet" if all(e is not None and e < 100 for e in embed) else "loaded" if min(e or 0 for e in embed) >= 300 else "MIXED"
print(f"LEAN {arm} pooled {res['pooled']:.3f} prose {mp:.2f} code {mc:.2f} regime {regime} blocks {blocks} embed {embed}")
