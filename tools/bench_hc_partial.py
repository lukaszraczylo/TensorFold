"""Time GLM's decode hyper-connection partial kernel (hc.cu) and compare its output bit for bit with a saved reference.

    python tools/bench_hc_partial.py --rows 1,4,8,16 [--save ref.pt | --check ref.pt]

Random bf16 streams; each call cycles over several inputs and weight copies so L2 holds neither.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from tensorfold.families.glm5_next.cuda import glue

WIDE, MIX, NB = 16384, 24, 16
COPIES = 48                       # 48 x 0.79 MB of weights: more than L2 (24 MB), as between two uses in a real round


def make(rows: int, device, seed: int):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((COPIES, rows, WIDE), device=device, generator=g) * 0.5).to(torch.bfloat16)   # likewise
    fn = (torch.randn((COPIES, MIX, WIDE), device=device, generator=g) * 0.02).to(torch.bfloat16)
    part = torch.zeros((rows, NB, 32), dtype=torch.float32, device=device)
    return x, fn, part


def time_rows(rows: int, device, reps: int, per_graph: int) -> dict:
    ext = glue._hc_ext()
    x, fn, part = make(rows, device, 1)
    for i in range(COPIES):
        ext.hc_partial(x[i], fn[i], part, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(per_graph):
            ext.hc_partial(x[i % COPIES], fn[i % COPIES], part, rows)
    graph.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    us = []
    for _ in range(reps):
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        us.append(start.elapsed_time(stop) * 1000 / per_graph)
    return {"rows": rows, "us_median": round(statistics.median(us), 2), "us_min": round(min(us), 2)}


def snapshot(rows: int, device) -> torch.Tensor:
    ext = glue._hc_ext()
    x, fn, part = make(rows, device, 5)
    ext.hc_partial(x[0], fn[0], part, rows)
    torch.cuda.synchronize()
    return part.clone().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,4,8,16")
    ap.add_argument("--reps", type=int, default=41)
    ap.add_argument("--per-graph", type=int, default=24)
    ap.add_argument("--save")
    ap.add_argument("--check")
    args = ap.parse_args()
    device = torch.device("cuda")
    rows = [int(v) for v in args.rows.split(",")]
    if args.save or args.check:
        snaps = {str(r): snapshot(r, device) for r in rows}
        if args.save:
            torch.save(snaps, args.save)
            print(f"saved {len(snaps)} outputs")
        else:
            ref = torch.load(args.check)
            bad = [k for k in snaps if not torch.equal(snaps[k], ref[k])]
            print("BITS-EQUAL" if not bad else f"BITS-DIFFER {bad}")
    for r in rows:
        print(json.dumps(time_rows(r, device, args.reps, args.per_graph)), flush=True)


if __name__ == "__main__":
    main()
