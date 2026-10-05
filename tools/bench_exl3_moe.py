"""Time the GLM-5.3-Flash EXL3 routed-expert decode path (route, rotate, gate/up, epilogue, down) on synthetic weights.

    python tools/bench_exl3_moe.py --rows 1,4,8,16 --distinct 24,28,32 [--save ref.pt | --check ref.pt]

One rank's layer shape (D 4096, NI 1024, 288 routed experts and the shared one). A decode kernel streams weights, so the
trellis words can be random: the time depends on how many distinct experts a window touches, not on their values.
Windows cycle over several expert pools so L2 (24 MB on GB10) never holds a pool's weights, and run in a CUDA graph
so launch gaps match the engine's. ``--save`` writes the outputs of fixed windows; ``--check`` compares them bit for
bit, the gate for a kernel change that must keep drafted replies equal to serial ones.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.families.glm5_next.cuda import exl3_mm as mm

D, NI, ROUTED, SLOTS = 4096, 1024, 288, 9
LIMIT = 7.0
POOLS = 4
PAIR_BYTES = 0


def build(device: torch.device, seed: int) -> mm.Exl3Experts:
    g = torch.Generator(device=device).manual_seed(seed)
    E = ROUTED + 1

    def words(*shape: int) -> torch.Tensor:
        return torch.randint(
            -(2**31), 2**31 - 1, shape, dtype=torch.int32, device=device, generator=g
        )

    def scales(*shape: int) -> torch.Tensor:
        return (torch.randn(shape, device=device, generator=g) * 0.02).to(torch.float16)

    suh = scales(E, D)
    return mm.Exl3Experts(
        gt=words(E, D // 16, NI // 16, 32),
        ut=words(E, D // 16, NI // 16, 32),
        dt=words(E, NI // 16, D // 16, 32),
        suh_g=suh,
        suh_u=suh.clone(),
        svh_g=scales(E, NI),
        svh_u=scales(E, NI),
        suh_d=scales(E, NI),
        svh_d=scales(E, D),
        count=E,
        width=NI,
        dims=D,
    )


def picks_for(
    rows: int, distinct: int, pool: int, device: torch.device
) -> torch.Tensor:
    """[rows, 9] int32: eight routed picks a row drawn from ``distinct`` experts in all, the shared expert last."""

    if distinct > ROUTED or distinct < SLOTS - 1:
        raise ValueError(
            f"distinct experts {distinct}: between {SLOTS - 1} and {ROUTED}"
        )
    order = torch.randperm(
        ROUTED, generator=torch.Generator().manual_seed(1000 + pool)
    )[:distinct]
    p = torch.arange(rows * (SLOTS - 1)) % distinct
    routed = order[p].reshape(rows, SLOTS - 1)
    shared = torch.full((rows, 1), ROUTED)
    return torch.cat([routed, shared], dim=1).to(torch.int32).to(device).contiguous()


def expert_bytes(ex: mm.Exl3Experts) -> int:
    return sum(t[0].numel() * t.element_size() for t in (ex.gt, ex.ut, ex.dt))


def run_window(ex, plan, scratch, x, picks, y, rows):
    grouped.route(picks, plan)
    mm.routed(x, picks, plan, ex, scratch, y, rows, LIMIT)


def time_graph(ex, rows: int, distinct: int, device, reps: int, per_graph: int) -> dict:
    plan = grouped.Plan(rows, SLOTS, ROUTED + 1, device)
    scratch = mm.Scratch(rows, SLOTS, D, NI, device)
    x = (torch.randn((rows, D), device=device) * 0.1).to(torch.bfloat16)
    y = torch.zeros((rows * SLOTS, D), dtype=torch.float32, device=device)
    sets = [picks_for(rows, distinct, k, device) for k in range(POOLS)]
    for p in sets:
        run_window(ex, plan, scratch, x, p, y, rows)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(per_graph):
            run_window(ex, plan, scratch, x, sets[i % POOLS], y, rows)
    graph.replay()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    us = []
    for _ in range(reps):
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        us.append(start.elapsed_time(stop) * 1000 / per_graph)
    seen = int(plan.counts[1].item())
    moved = distinct_bytes(sets[0], ex)
    best, med = min(us), statistics.median(us)
    return {
        "rows": rows,
        "distinct": distinct,
        "plan_distinct": seen,
        "us_median": round(med, 1),
        "us_min": round(best, 1),
        "gb_s_median": round(moved / med / 1e3, 1),
        "mb_moved": round(moved / 1e6, 1),
    }


def distinct_bytes(picks: torch.Tensor, ex: mm.Exl3Experts) -> int:
    return int(torch.unique(picks).numel()) * expert_bytes(ex)


def snapshot(ex, device, rows: int, distinct: int) -> torch.Tensor:
    plan = grouped.Plan(rows, SLOTS, ROUTED + 1, device)
    scratch = mm.Scratch(rows, SLOTS, D, NI, device)
    x = (
        torch.randn(
            (rows, D),
            device=device,
            generator=torch.Generator(device=device).manual_seed(7),
        )
        * 0.1
    )
    y = torch.zeros((rows * SLOTS, D), dtype=torch.float32, device=device)
    run_window(
        ex,
        plan,
        scratch,
        x.to(torch.bfloat16),
        picks_for(rows, distinct, 0, device),
        y,
        rows,
    )
    torch.cuda.synchronize()
    return y.clone().cpu()


def kernel_table(ex, rows: int, distinct: int, device) -> list[dict]:
    from torch.profiler import ProfilerActivity, profile

    plan = grouped.Plan(rows, SLOTS, ROUTED + 1, device)
    scratch = mm.Scratch(rows, SLOTS, D, NI, device)
    x = (torch.randn((rows, D), device=device) * 0.1).to(torch.bfloat16)
    y = torch.zeros((rows * SLOTS, D), dtype=torch.float32, device=device)
    sets = [picks_for(rows, distinct, k, device) for k in range(POOLS)]
    for p in sets:
        run_window(ex, plan, scratch, x, p, y, rows)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(40):
            run_window(ex, plan, scratch, x, sets[i % POOLS], y, rows)
        torch.cuda.synchronize()
    out = []
    for e in prof.key_averages():
        if e.device_time_total > 0:
            out.append(
                {
                    "kernel": e.key[:48],
                    "calls": e.count,
                    "us_avg": round(e.device_time_total / e.count, 1),
                }
            )
    return sorted(out, key=lambda r: -r["us_avg"] * r["calls"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,4,8,16")
    ap.add_argument("--distinct", default="24,28,32")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--per-graph", type=int, default=24)
    ap.add_argument(
        "--kernels",
        action="store_true",
        help="per-kernel time table for the first rows/distinct pair",
    )
    ap.add_argument(
        "--oracle-grid",
        action="store_true",
        help="launch exactly the live items (an upper bound for a compact grid)",
    )
    ap.add_argument("--save")
    ap.add_argument("--check")
    args = ap.parse_args()
    device = torch.device("cuda")
    ex = build(device, 0)
    rows = [int(v) for v in args.rows.split(",")]
    distinct = [int(v) for v in args.distinct.split(",")]
    if args.save or args.check:
        snaps = {
            f"{r}x{u}": snapshot(ex, device, r, u)
            for r in rows
            for u in distinct
            if u >= SLOTS - 1
        }
        if args.save:
            torch.save(snaps, args.save)
            print(f"saved {len(snaps)} windows to {args.save}")
        else:
            ref = torch.load(args.check)
            bad = [k for k in snaps if not torch.equal(snaps[k], ref[k])]
            print("BITS-EQUAL" if not bad else f"BITS-DIFFER {bad}")
    for r in rows:
        for u in distinct:
            print(
                json.dumps(time_graph(ex, r, u, device, args.reps, args.per_graph)),
                flush=True,
            )
    if args.kernels:
        for row in kernel_table(ex, rows[0], distinct[0], device)[:10]:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
