#!/usr/bin/env bash
# trial.sh <name> ["K=V K2=V2"] [src-dir]   restart the dev server (optionally with env overrides / a source overlay),
# then run lean_bench (20 pairs). Prints the LEAN line; log in /tmp/glm-trials/<name>.out
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); name=$1; export TRIAL_ENV="TF_GLM_DFLASH_POLICY=fnc7:0.4 ${2:-}"; mkdir -p /tmp/glm-trials
[ -n "${3:-}" ] && "$D/devctl.sh" sync "$3"
"$D/devctl.sh" stop >/dev/null 2>&1 || true
"$D/devctl.sh" start >/dev/null
"$D/devctl.sh" wait 1500 | tail -1
python3 "$D/lean_bench.py" "$name" http://10.0.1.110:8888 "/tmp/glm-trials/$name.json" | tee "/tmp/glm-trials/$name.out"
