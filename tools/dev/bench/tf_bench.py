#!/usr/bin/env python3
"""tf_bench.py runs quiet_bench.py's protocol on a server without vLLM metrics.

TensorFold has no /metrics route, so the speculative-decoding counters and
request_success_total that quiet_bench.py reads do not exist. This script
reuses the pinned upstream harness (stream_bench, prompt sets, tok/s formula)
and quiet_bench's embedder gate, and drops only those counters. The tok/s
numbers stay comparable with quiet_bench.py runs. Acceptance is not measured.

Usage: tf_bench.py <arm> <base-url> <out-dir> [sets]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import drafter_ab as ab
import quiet_bench as qb


def one_run(
    bd: Any,
    prompt: str,
    thinking: bool | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Run one streamed request and return its record, void on any problem."""
    orig = bd._post
    if thinking is None:
        bd._post = ab.strip_kwargs_post(orig)
    try:
        r = bd.stream_bench(max_tokens=max_tokens, prompt=prompt)
    finally:
        bd._post = orig
    problems: list[str] = []
    if r["http"] != 200:
        problems.append(f"http {r['http']}")
    if r["completion_tokens"] != max_tokens:
        problems.append(
            f"completion_tokens {r['completion_tokens']} != {max_tokens}",
        )
    if not r["text"]:
        problems.append("empty text")
    if r["nan"]:
        problems.append("nan in text")
    return {
        "tok_s": r["tok_s"],
        "ttft_s": r["ttft_s"],
        "completion_tokens": r["completion_tokens"],
        "finish_reason": r["finish_reason"],
        "void": bool(problems),
        "problems": problems,
        "text_head": r["text_head"][:160],
    }


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
    while sum(1 for r in runs if not r["discard"] and not r["void"]) < qb.RUNS:
        attempts += 1
        if attempts > qb.MAX_ATTEMPTS:
            raise SystemExit(
                f"{name}: {qb.MAX_ATTEMPTS} attempts, too few runs",
            )
        before = qb.wait_quiet(deadline)
        r = one_run(bd, prompt, thinking, qb.MAX_TOKENS)
        after = qb.embed_m()
        r["embed_m"] = [before, after]
        if after is None or after >= qb.QUIET_M:
            r["void"] = True
            r["problems"].append(f"embed busy after run ({after}m)")
        kept = [x for x in runs if not x["void"]]
        r["discard"] = len(kept) < qb.DISCARD and not r["void"]
        runs.append(r)
        tag = (
            "DISCARD"
            if r["discard"]
            else ("VOID " + "; ".join(r["problems"]) if r["void"] else "ok")
        )
        line = f"{arm} {name} #{len(runs) - 1}: {r['tok_s']:.2f} tok/s {tag}"
        print(f"{line} embed {before}->{after}m", flush=True)
        time.sleep(qb.GAP_S)
    return {"thinking": thinking, "runs": runs, "attempts": attempts}


def main() -> int:
    """Run one arm and write its JSON record."""
    arm, base, out_dir = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]
    chosen = sys.argv[4] if len(sys.argv) > 4 else "prose,code"
    bd, digest = ab.load_harness(qb.UPSTREAM, False)
    sets = ab.prompt_sets(bd)
    bd.BASE = base
    code, body = bd.health()
    if code != 200:
        raise SystemExit(f"/health returned {code}: {body[:200]}")
    deadline = time.time() + qb.MAX_WAIT_S
    result: dict[str, Any] = {
        "arm": arm,
        "base": base,
        "model": bd.MODEL,
        "started": ab.utc_now(),
        "harness": {"sha256": digest, "pinned_sha256": ab.HARNESS_SHA256},
        "protocol": {
            "runs": qb.RUNS,
            "discard": qb.DISCARD,
            "max_tokens": qb.MAX_TOKENS,
            "temperature": 0,
            "quiet_millicores": qb.QUIET_M,
            "spec_counters": "not available",
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
    result["finished"] = ab.utc_now()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"tf-bench-{arm}.json"
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
