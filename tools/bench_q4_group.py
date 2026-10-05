"""Time the sm_12x 4-bit lane matmul (qmm_group) on GLM-5.3-Flash's dense shapes with random weights, in CUDA graphs.

    python tools/bench_q4_group.py --rows 1,6,16 --tiles 0,2,8 [--shapes qkv,o_kda,...] [--check ref.pt | --save ref.pt]

Note: the GLM forward calls ``qmm.matmul`` on stacked weights (``families/glm5_next/cuda/qmm.py``), not
``matmul_group``, so per-shape figures here are for the grouped kernel alone, not the production launch path.
Weights are random: the kernel streams them, so only the shape matters. Each shape cycles over enough copies to
exceed L2, so figures are DRAM GB/s of weight, scale and bias bytes. A tile never changes bits (``--save`` and
``--check`` compare outputs bit for bit); ``--sk`` overrides the K slices (which does change bits).
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from tensorfold.cuda.kernels import qmm

GS = 64
# name: (parts as (n, k) per projection of one input) - one rank's share of GLM-5.3-Flash, TP 2
SHAPES = {
    "qkv": [(4096, 4096)] * 3,
    "o_kda": [(4096, 4096)],
    "fg_a": [(128, 4096)] * 2,
    "fg_b": [(4096, 128)] * 2,
    "shared_gu": [(1024, 4096)] * 2,
    "shared_d": [(4096, 1024)],
    "dense_gu": [(6144, 4096)] * 2,
    "dense_d": [(4096, 6144)],
    "dsa_qa": [(1536, 4096)],
    "dsa_qb": [(8192, 1536)],
    "dsa_kvb": [(16384, 512)],
    "dsa_o": [(4096, 8192)],
}
COPIES_BYTES = 192 * 1024 * 1024


def rand_q4(n: int, k: int, device) -> qmm.Q4:
    npad, kg = -(-n // 128) * 128, k // GS
    weight = torch.randint(-(2**31), 2**31 - 1, (npad // 64, kg, 8, 32, GS // 32), dtype=torch.int32, device=device)
    scales = (torch.randn((kg, npad), device=device) * 0.01).to(torch.bfloat16)
    biases = (torch.randn((kg, npad), device=device) * 0.01).to(torch.bfloat16)
    return qmm.Q4(weight, scales, biases, n, k, GS)


def part_bytes(q: qmm.Q4) -> int:
    return q.nbytes()


def time_shape(name: str, parts, rows: int, tile: int, sk: int | None, device, reps: int, per_graph: int) -> dict:
    one = sum(n * k * 0.5625 for n, k in parts)
    copies = max(2, int(COPIES_BYTES // one) + 1)
    sets = [[rand_q4(n, k, device) for n, k in parts] for _ in range(copies)]
    k_in = parts[0][1]
    x = (torch.randn((rows, k_in), device=device) * 0.1).to(torch.bfloat16)
    xs = qmm.group_sums(x, GS)
    sks = None if sk is None else [sk] * len(parts)

    def call(i: int):
        return qmm.matmul_group(x, sets[i % copies], xs, tile=tile, sks=sks)

    for i in range(copies):
        call(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(per_graph):
            call(i)
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
    med = statistics.median(us)
    nbytes = sum(part_bytes(q) for q in sets[0])
    return {"shape": name, "rows": rows, "tile": tile, "sk": sk, "us": round(med, 1), "gb_s": round(nbytes / med / 1e3, 1),
            "mb": round(nbytes / 1e6, 2)}


def snapshot(parts, rows: int, tile: int, device) -> list[torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(3)
    qs = [rand_q4(n, k, device) for n, k in parts]
    x = (torch.randn((rows, parts[0][1]), device=device, generator=g) * 0.1).to(torch.bfloat16)
    outs = qmm.matmul_group(x, qs, qmm.group_sums(x, GS), tile=tile)
    torch.cuda.synchronize()
    return [o.float().cpu() for o in outs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,6,16")
    ap.add_argument("--tiles", default="0")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--sk", type=int)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--per-graph", type=int, default=16)
    ap.add_argument("--save")
    ap.add_argument("--check")
    args = ap.parse_args()
    device = torch.device("cuda")
    names = args.shapes.split(",")
    rows = [int(v) for v in args.rows.split(",")]
    tiles = [int(v) for v in args.tiles.split(",")]
    if args.save or args.check:
        torch.manual_seed(0)
        snaps = {f"{s}:{r}:{t}": snapshot(SHAPES[s], r, t, device) for s in names for r in rows for t in tiles}
        if args.save:
            torch.save(snaps, args.save)
            print(f"saved {len(snaps)} outputs")
        else:
            ref = torch.load(args.check)
            bad = [k for k in snaps if any(not torch.equal(a, b) for a, b in zip(snaps[k], ref[k]))]
            print("BITS-EQUAL" if not bad else f"BITS-DIFFER {bad[:6]}")
    for s in names:
        for r in rows:
            for t in tiles:
                print(json.dumps(time_shape(s, SHAPES[s], r, t, args.sk, device, args.reps, args.per_graph)), flush=True)


if __name__ == "__main__":
    main()
