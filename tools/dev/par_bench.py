#!/usr/bin/env python3
"""par_bench.py <arm> <base-url> <out.json> <streams> [rounds=6]

N concurrent streamed requests (prose, code, prose, code ...), 400 tokens, temperature 0, thinking off, through the
pinned upstream harness. Reports per-request and aggregate tok/s per round, and whether every reply equals the same
prompt run alone. Round 0 is a discard.
"""
import json
import statistics as st
import sys
import threading
import time

BENCH = "/Users/nvm/Documents/projects/home-cluster/.wt-dev/namespaces/llm/base/glm53/bench"
sys.path.insert(0, BENCH)
import drafter_ab as ab  # noqa: E402
import quiet_bench as qb  # noqa: E402

arm, base, out, n = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3], int(sys.argv[4])
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 6
bd, _ = ab.load_harness(qb.UPSTREAM, False)
bd.BASE = base
sets = ab.prompt_sets(bd)
names = ["prose", "code"]
MT = qb.MAX_TOKENS


def one(prompt, res, i):
    t0 = time.perf_counter()
    r = bd.stream_bench(max_tokens=MT, prompt=prompt)
    res[i] = (r, t0, time.perf_counter())


alone = {}
for nm in names:
    res = {}
    one(sets[nm][0], res, 0)
    alone[nm] = res[0][0]["text"]
    print(f"alone {nm}: {res[0][0]['tok_s']:.2f} tok/s", flush=True)

recs = []
for k in range(rounds + 1):
    res, th = {}, []
    for i in range(n):
        th.append(threading.Thread(target=one, args=(sets[names[i % 2]][0], res, i)))
    for t in th:
        t.start()
    for t in th:
        t.join()
    toks = sum(res[i][0]["completion_tokens"] for i in range(n))
    wall = max(res[i][2] for i in range(n)) - min(res[i][1] for i in range(n))
    per = [res[i][0]["tok_s"] for i in range(n)]
    same = [res[i][0]["text"] == alone[names[i % 2]] for i in range(n)]
    rec = {"round": k, "discard": k == 0, "agg_tok_s": toks / wall, "per_req": per,
           "ttft_s": [res[i][0]["ttft_s"] for i in range(n)], "identical_to_alone": same}
    recs.append(rec)
    print(f"{arm} n={n} #{k}{' DISCARD' if k == 0 else ''}: agg {rec['agg_tok_s']:.1f} per-req "
          f"{[round(p, 1) for p in per]} identical {same}", flush=True)
kept = [r for r in recs if not r["discard"]]
summary = {"agg_median": st.median(r["agg_tok_s"] for r in kept),
           "per_req_median": st.median(p for r in kept for p in r["per_req"]),
           "all_identical": all(all(r["identical_to_alone"]) for r in kept)}
print("SUMMARY", json.dumps(summary), flush=True)
json.dump({"arm": arm, "n": n, "summary": summary, "rounds": recs}, open(out, "w"), indent=1)
