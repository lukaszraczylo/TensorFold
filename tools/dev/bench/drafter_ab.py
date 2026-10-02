#!/usr/bin/env python3
"""Decode A/B bench for the GLM-5.3 LeaderWorkerSet on the GX10 pair.

Drives the live OpenAI API of the leader pod and records, per run, decode
tok/s and the spec-decode counter deltas (drafts, draft tokens, accepted
tokens, per-position acceptance). Written for the 2026-09-10 DFlash2 drafter
A/B. The record is ../drafter-ab-20260910.json, and drafter_compare.py gives
the verdict.

The request body, the tok/s formula and the counter parsing come from
upstream's tests/bench_decode.py. The script imports that file unedited from
a MiaAI-Lab clone, so the numbers stay comparable with
../decode-tps-20260909.json. It checks the file's sha256 first and refuses a
drifted copy unless --allow-harness-drift is given.

Primary metric of that A/B: pooled accepted tokens per draft step,
sum(accepted) / sum(drafts), over the prose and code sets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

UPSTREAM_COMMIT: Final = "1caea9a"
HARNESS_RELPATH: Final = Path("tests") / "bench_decode.py"
# Unchanged from upstream 4bac6e2 (2026-09-07) through 1caea9a (2026-09-10).
HARNESS_SHA256: Final = (
    "0dccf0d49ec9f19df3459edacd599097ca7c4a9d506e6665fc70c1ff958f4a8d"
)

CODE_PROMPT: Final = (
    "Write a complete, idiomatic Go implementation of a generic LRU cache "
    "with a mutex, a doubly linked list, Get/Put/Len methods and "
    "table-driven tests."
)
DEFAULT_SETS: Final = "prose,code,structured,prose_thinking_default"
PRIMARY_SETS: Final = ("prose", "code")
IDLE_WAIT_S: Final = 180
IDLE_POLL_S: Final = 5
METRIC_PREFIXES: Final = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:request_success_total",
)
SUMMARY_KEYS: Final = (
    "n_valid",
    "n_void",
    "tok_s_median",
    "tok_s_sd",
    "pooled_accepted_per_step",
    "pooled_accept_ratio",
)

PROTOCOL_HARNESS: Final = (
    "upstream tests/bench_decode.py stream_bench + spec_delta, unedited"
)

PostFn = Callable[..., Any]


def load_harness(upstream: Path, allow_drift: bool) -> tuple[Any, str]:
    """Import upstream tests/bench_decode.py after checking its sha256.

    Args:
        upstream: Path to a MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks clone.
        allow_drift: Import the file even when its sha256 differs.

    Returns:
        The imported module and the sha256 of the file.

    Raises:
        SystemExit: The file is missing, drifted, or cannot be imported.
    """
    path = upstream / HARNESS_RELPATH
    if not path.is_file():
        raise SystemExit(
            f"{path} not found. Clone "
            "MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks and check out "
            f"{UPSTREAM_COMMIT}; see README.md.",
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != HARNESS_SHA256 and not allow_drift:
        raise SystemExit(
            f"{path} sha256 {digest} != {HARNESS_SHA256}: the harness "
            "changed, so new numbers would not compare with the recorded "
            f"ones. Check out {UPSTREAM_COMMIT} or pass "
            "--allow-harness-drift deliberately.",
        )
    spec = importlib.util.spec_from_file_location("bench_decode", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, digest


def prompt_sets(bd: Any) -> dict[str, tuple[str, bool | None]]:
    """Return the prompt sets, keyed by set name.

    Args:
        bd: The imported upstream harness module.

    Returns:
        Set name -> (prompt, thinking). thinking=None sends no
        chat_template_kwargs, so the server default applies (thinking on,
        effort Max), which is the production path.
    """
    return {
        "prose": (bd.BENCH_PROMPT, False),
        "code": (CODE_PROMPT, False),
        "structured": (bd.STRUCTURED_PROMPT, False),
        "prose_thinking_default": (bd.BENCH_PROMPT, None),
    }


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string, in seconds."""
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def metrics(base: str) -> dict[str, float]:
    """Sum the load and request-count gauges from the vLLM /metrics page.

    Args:
        base: Leader URL without a trailing slash.

    Returns:
        Metric name -> value, summed over label sets.
    """
    with urllib.request.urlopen(base + "/metrics", timeout=10) as resp:
        raw = resp.read().decode()
    out: dict[str, float] = {}
    for line in raw.splitlines():
        if line.startswith(METRIC_PREFIXES):
            name = line.split("{")[0].split()[0]
            out[name] = out.get(name, 0.0) + float(line.split()[-1])
    return out


def wait_idle(base: str) -> float:
    """Block until no request is running or waiting on the endpoint.

    Args:
        base: Leader URL without a trailing slash.

    Returns:
        Seconds spent waiting.

    Raises:
        SystemExit: The endpoint stayed busy for IDLE_WAIT_S seconds.
    """
    t0 = time.time()
    while time.time() - t0 < IDLE_WAIT_S:
        m = metrics(base)
        running = m.get("vllm:num_requests_running", 0)
        waiting = m.get("vllm:num_requests_waiting", 0)
        if running + waiting == 0:
            return time.time() - t0
        time.sleep(IDLE_POLL_S)
    raise SystemExit(
        f"endpoint not idle after {IDLE_WAIT_S}s; aborting rather than "
        "measuring under load",
    )


def strip_kwargs_post(orig: PostFn) -> PostFn:
    """Wrap the harness POST function so it drops chat_template_kwargs.

    Args:
        orig: The harness's own _post function.

    Returns:
        A drop-in replacement, so the server default applies instead.
    """

    def _post(
        path: str,
        body: dict,
        timeout: float = 600.0,
        stream: bool = False,
    ) -> Any:
        body = dict(body)
        body.pop("chat_template_kwargs", None)
        return orig(path, body, timeout=timeout, stream=stream)

    return _post


def one_run(
    bd: Any,
    prompt: str,
    thinking: bool | None,
    max_tokens: int,
) -> dict[str, Any]:
    """Run one streamed request and check that it measured what it claims.

    Args:
        bd: The imported upstream harness module, with BASE set.
        prompt: The user message.
        thinking: None sends the server default. Any other value keeps the
            harness body, which sets enable_thinking to false.
        max_tokens: Completion length. A shorter completion voids the run.

    Returns:
        The run record: tok/s, TTFT, spec-decode deltas, and a void flag
        with its reasons.
    """
    orig = bd._post
    if thinking is None:
        bd._post = strip_kwargs_post(orig)
    try:
        wait_s = wait_idle(bd.BASE)
        ok_before = metrics(bd.BASE).get("vllm:request_success_total", 0)
        s0 = bd.spec_snapshot()
        r = bd.stream_bench(max_tokens=max_tokens, prompt=prompt)
        s1 = bd.spec_snapshot()
        ok_after = metrics(bd.BASE).get("vllm:request_success_total", 0)
    finally:
        bd._post = orig
    spec = bd.spec_delta(s0, s1)
    other = int(ok_after - ok_before) - 1
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
    if other != 0:
        problems.append(f"{other} other request(s) finished during the run")
    if not spec["drafts"]:
        problems.append("no spec-decode drafts counted")
    return {
        "tok_s": r["tok_s"],
        "ttft_s": r["ttft_s"],
        "completion_tokens": r["completion_tokens"],
        "finish_reason": r["finish_reason"],
        "idle_wait_s": round(wait_s, 1),
        "spec": spec,
        "void": bool(problems),
        "problems": problems,
        "text_head": r["text_head"][:160],
    }


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise the measured runs of one set.

    Args:
        runs: Run records from one_run, with the discards already removed.

    Returns:
        Run counts, the tok/s median and sd, and the pooled acceptance.
    """
    ok = [r for r in runs if not r["void"]]
    tps = [r["tok_s"] for r in ok if r["tok_s"]]
    acc = sum(r["spec"]["accepted"] for r in ok)
    drafts = sum(r["spec"]["drafts"] for r in ok)
    dtok = sum(r["spec"]["draft_tokens"] for r in ok)
    median = round(statistics.median(tps), 2) if tps else None
    sd = round(statistics.stdev(tps), 2) if len(tps) > 1 else None
    per_step = round(acc / drafts, 4) if drafts else None
    ratio = round(acc / dtok, 4) if dtok else None
    return {
        "n_valid": len(ok),
        "n_void": len(runs) - len(ok),
        "tok_s_median": median,
        "tok_s_sd": sd,
        "tok_s_runs": [round(x, 2) for x in tps],
        "pooled_accepted_per_step": per_step,
        "pooled_accept_ratio": ratio,
        "accepted_per_step_runs": [r["spec"]["accepted_per_step"] for r in ok],
        "drafts": drafts,
        "accepted": acc,
        "draft_tokens": dtok,
    }


def primary_pooled(result: dict[str, Any]) -> float | None:
    """Pool accepted tokens per draft step over the primary sets.

    Args:
        result: The arm record, with its sets filled in.

    Returns:
        sum(accepted) / sum(drafts) over the valid measured runs, or None
        when no primary set ran.
    """
    prim = [
        r
        for s in PRIMARY_SETS
        if s in result["sets"]
        for r in result["sets"][s]["runs"]
        if not r["discard"] and not r["void"]
    ]
    drafts = sum(r["spec"]["drafts"] for r in prim)
    if not drafts:
        return None
    return round(sum(r["spec"]["accepted"] for r in prim) / drafts, 4)


def run_line(arm: str, name: str, i: int, r: dict[str, Any]) -> str:
    """Format the progress line for one finished run."""
    if r["discard"]:
        flag = "DISCARD"
    elif r["void"]:
        flag = "VOID " + "; ".join(r["problems"])
    else:
        flag = "ok"
    tok_s = round(r["tok_s"], 2) if r["tok_s"] else None
    step = r["spec"]["accepted_per_step"]
    return f"{arm} {name} #{i}: {tok_s} tok/s L={step} {flag}"


def brief(result: dict[str, Any], out: Path) -> dict[str, Any]:
    """Return the short summary that is printed at the end of an arm."""
    return {
        "out": str(out),
        "primary": result["primary_pooled_accepted_per_step"],
        "summary": {
            name: {k: data["summary"][k] for k in SUMMARY_KEYS}
            for name, data in result["sets"].items()
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--upstream",
        required=True,
        type=Path,
        help=f"MiaAI-Lab clone checked out at {UPSTREAM_COMMIT}",
    )
    ap.add_argument(
        "--allow-harness-drift",
        action="store_true",
        help="run even when tests/bench_decode.py differs from the pin",
    )
    ap.add_argument(
        "--arm",
        required=True,
        help="label for this arm, for example the drafter revision",
    )
    ap.add_argument(
        "--base",
        required=True,
        help="leader pod URL, never a port-forward",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="served model name to send instead of the harness default "
        "(GLM-5.3-Flash-EXL3), for example DeepSeek-v4.1-Flash-EXL3",
    )
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--discard", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--sets", default=DEFAULT_SETS)
    ap.add_argument("--out-dir", type=Path, default=Path.cwd())
    return ap.parse_args()


def main() -> int:
    """Run one arm and write its JSON record.

    Returns:
        The process exit status.
    """
    args = parse_args()
    upstream = args.upstream.expanduser()
    bd, digest = load_harness(upstream, args.allow_harness_drift)
    sets = prompt_sets(bd)
    chosen = args.sets.split(",")
    unknown = [s for s in chosen if s not in sets]
    if unknown:
        raise SystemExit(
            f"unknown set(s) {unknown}; choose from {list(sets)}",
        )
    bd.BASE = args.base.rstrip("/")
    if args.model:
        bd.MODEL = args.model
    code, body = bd.health()
    if code != 200:
        raise SystemExit(f"/health returned {code}: {body[:200]}")

    started = utc_now()
    result: dict[str, Any] = {
        "arm": args.arm,
        "base": bd.BASE,
        "model": bd.MODEL,
        "started": started,
        "harness": {
            "file": str(HARNESS_RELPATH),
            "sha256": digest,
            "pinned_sha256": HARNESS_SHA256,
            "pinned_upstream_commit": UPSTREAM_COMMIT,
        },
        "protocol": {
            "runs": args.runs,
            "discard": args.discard,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "gap_s": args.gap,
            "harness": PROTOCOL_HARNESS,
        },
        "sets": {},
    }
    for name in chosen:
        prompt, thinking = sets[name]
        runs = []
        for i in range(args.discard + args.runs):
            r = one_run(bd, prompt, thinking, args.max_tokens)
            r["discard"] = i < args.discard
            runs.append(r)
            print(run_line(args.arm, name, i, r), flush=True)
            time.sleep(args.gap)
        measured = [r for r in runs if not r["discard"]]
        result["sets"][name] = {
            "thinking": thinking,
            "summary": summarise(measured),
            "runs": runs,
        }

    result["primary_pooled_accepted_per_step"] = primary_pooled(result)
    result["finished"] = utc_now()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started[:19].replace(":", "")
    out = args.out_dir / f"drafter-ab-{args.arm}-{stamp}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(brief(result, out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
