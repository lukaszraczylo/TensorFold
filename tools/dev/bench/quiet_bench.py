#!/usr/bin/env python3
"""drafter_ab.py runs, voided per run when the co-resident embedder is busy.

The embedder on the GX10s bursts every few minutes, and a busy embedder
slows the TP=2 step by about 25 percent (.claude/rules/gx10-benchmarking.md).
A gate at the start of a bench is not enough: on 2026-09-18 it opened at
17:44:56 and the embedder ran at 305-686m from 17:47 for the rest of the run.

This wraps drafter_ab.py's own functions unchanged (harness sha check,
prompt sets, one_run, summarise, primary_pooled). Before each run it waits
until the busiest embed pod is under QUIET_M millicores. After the run it
checks again, and a busy reading voids that run. It continues until RUNS
valid measured runs per set, after DISCARD warm runs.

kubectl top averages over about a minute, so a burst inside a short run can
still pass both checks. Compare medians, and check the step rate
(tok/s divided by accepted-per-step + 1) for outliers before a verdict.

Usage: quiet_bench.py <arm> <base-url> <out-dir> [sets]
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import drafter_ab as ab

UPSTREAM = Path(
    "/Users/nvm/Documents/projects/private/dgxspark/glm53-upstream",
)
CTX = ["kubectl", "--context", "admin@home", "-n", "llm"]
QUIET_M = 100
RUNS = 8
DISCARD = 1
MAX_TOKENS = 400
GAP_S = 2.0
MAX_ATTEMPTS = 40
MAX_WAIT_S = 3 * 3600


def embed_m() -> int | None:
    """Return the busiest embed pod's CPU in millicores, or None."""
    cmd = [*CTX, "get", "pods", "--no-headers"]
    cmd += ["-o", "custom-columns=N:.metadata.name"]
    names = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.split()
    vals = []
    for name in names:
        if not name.startswith("embed-llamacpp"):
            continue
        out = subprocess.run(
            [*CTX, "top", "pod", name, "--no-headers"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.split()
        if len(out) >= 2 and out[1].endswith("m"):
            vals.append(int(out[1][:-1]))
    return max(vals) if vals else None


def wait_quiet(deadline: float) -> int:
    """Block until the embedder is quiet, and return its reading."""
    while True:
        m = embed_m()
        if m is not None and m < QUIET_M:
            return m
        if time.time() > deadline:
            raise SystemExit("embedder never went quiet before the deadline")
        time.sleep(20)


def run_set(
    bd: Any,
    arm: str,
    name: str,
    prompt: str,
    thinking: bool | None,
    deadline: float,
) -> dict[str, Any]:
    """Collect RUNS valid measured runs of one prompt set."""
    runs: list[dict[str, Any]] = []
    attempts = 0
    while sum(1 for r in runs if not r["discard"] and not r["void"]) < RUNS:
        attempts += 1
        if attempts > MAX_ATTEMPTS:
            raise SystemExit(f"{name}: {MAX_ATTEMPTS} attempts, too few runs")
        before = wait_quiet(deadline)
        r = ab.one_run(bd, prompt, thinking, MAX_TOKENS)
        after = embed_m()
        r["embed_m"] = [before, after]
        if after is None or after >= QUIET_M:
            r["void"] = True
            r["problems"].append(f"embed busy after run ({after}m)")
        kept = [x for x in runs if not x["void"]]
        r["discard"] = len(kept) < DISCARD and not r["void"]
        runs.append(r)
        line = ab.run_line(arm, name, len(runs) - 1, r)
        print(line, f"embed {before}->{after}m", flush=True)
        time.sleep(GAP_S)
    measured = [r for r in runs if not r["discard"]]
    return {
        "thinking": thinking,
        "summary": ab.summarise(measured),
        "runs": runs,
        "attempts": attempts,
    }


def main() -> int:
    """Run one arm and write its JSON record."""
    arm, base, out_dir = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]
    chosen = "prose,code,structured"
    if len(sys.argv) > 4:
        chosen = sys.argv[4]
    bd, digest = ab.load_harness(UPSTREAM, False)
    sets = ab.prompt_sets(bd)
    bd.BASE = base
    code, body = bd.health()
    if code != 200:
        raise SystemExit(f"/health returned {code}: {body[:200]}")
    deadline = time.time() + MAX_WAIT_S
    result: dict[str, Any] = {
        "arm": arm,
        "base": base,
        "model": bd.MODEL,
        "started": ab.utc_now(),
        "harness": {"sha256": digest, "pinned_sha256": ab.HARNESS_SHA256},
        "protocol": {
            "runs": RUNS,
            "discard": DISCARD,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "gap_s": GAP_S,
            "quiet_millicores": QUIET_M,
            "void_rule": "embed busy before or after the run",
        },
        "sets": {},
    }
    for name in chosen.split(","):
        prompt, thinking = sets[name]
        result["sets"][name] = run_set(
            bd,
            arm,
            name,
            prompt,
            thinking,
            deadline,
        )
    result["primary_pooled_accepted_per_step"] = ab.primary_pooled(result)
    result["finished"] = ab.utc_now()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"quiet-bench-{arm}.json"
    path.write_text(json.dumps(result, indent=2))
    summary = {n: s["summary"] for n, s in result["sets"].items()}
    brief = {
        "out": str(path),
        "primary": result["primary_pooled_accepted_per_step"],
        "summary": summary,
    }
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
