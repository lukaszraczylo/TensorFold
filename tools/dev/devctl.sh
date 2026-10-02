#!/usr/bin/env bash
# devctl.sh sync <dir-containing-tensorfold/> | start | stop | wait [secs] | status | log <0|1> [lines]
# Drives the glm-tensorfold-dev LeaderWorkerSet (home-kubernetes namespaces/llm/base/tensorfold-dev-lws.yaml): the
# pods only sleep; this overlays source at /cache/dev/src and serves by hand. NSYS=1 ./devctl.sh start runs rank 0
# under `nsys launch` (session "prof").
set -euo pipefail
K=(kubectl --context admin@home -n llm)
PODS=(glm-tensorfold-dev-0 glm-tensorfold-dev-0-1)
SERVE='source /tmp/devenv.sh; for kv in $TRIAL_ENV; do export "$kv"; done; export PYTHONPATH=/cache/dev/src${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}
args=(serve "$MODEL_PATH" --tp 2 --rank "$RANK" --master "$MASTER" --master-port "$MASTER_PORT" --no-update-check
  --drafter "$DRAFTER" --context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" --thinking --reasoning-effort high --vision)
[ "$RANK" = 0 ] && args+=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT")
mkdir -p /cache/dev/log
find "${TORCH_EXTENSIONS_DIR:-/cache/torch_extensions}" -maxdepth 3 -type f -name lock -delete 2>/dev/null || true
echo "tensorfold ${args[*]}" > /cache/dev/log/rank$RANK.log
PRE="${TRIAL_PRE:-}"; [ -n "${NSYS:-}" ] && [ "$RANK" = 0 ] && PRE="nsys launch -t cuda,nvtx,osrt --cuda-graph-trace=node ${NSYS_FLAGS:-} --session-new=prof"
setsid nohup $PRE tensorfold "${args[@]}" >> /cache/dev/log/rank$RANK.log 2>&1 < /dev/null &
echo started rank $RANK'

cmd=${1:-status}
case "$cmd" in
  sync)
    src=${2:?dir containing tensorfold/}
    for p in "${PODS[@]}"; do
      "${K[@]}" exec "$p" -c tensorfold -- bash -c 'rm -rf /cache/dev/src && mkdir -p /cache/dev/src'
      tar -C "$src" --exclude='__pycache__' --exclude='*.pyc' -cf - tensorfold 2>/dev/null | "${K[@]}" exec -i "$p" -c tensorfold -- tar -C /cache/dev/src -xf - 2>/dev/null
    done
    echo "synced $src" ;;
  start)
    "${K[@]}" exec "${PODS[1]}" -c tensorfold -- bash -c "export NSYS=${NSYS:-} NSYS_FLAGS=\"${NSYS_FLAGS:-}\" TRIAL_ENV=\"${TRIAL_ENV:-}\" TRIAL_PRE=\"${TRIAL_PRE:-}\"; $SERVE"
    "${K[@]}" exec "${PODS[0]}" -c tensorfold -- bash -c "export NSYS=${NSYS:-} NSYS_FLAGS=\"${NSYS_FLAGS:-}\" TRIAL_ENV=\"${TRIAL_ENV:-}\" TRIAL_PRE=\"${TRIAL_PRE:-}\"; $SERVE" ;;
  stop)
    for p in "${PODS[@]}"; do
      "${K[@]}" exec "$p" -c tensorfold -- bash -c 'for s in $(nsys sessions list 2>/dev/null | awk "NR>1{print \$1}"); do nsys shutdown --session=$s 2>/dev/null || true; done; pkill -f "tensorfold serve" || true' >/dev/null 2>&1 || true
    done
    sleep 8 ;;
  wait)
    t=${2:-1500}; end=$((SECONDS+t))
    while (( SECONDS < end )); do
      if "${K[@]}" exec "${PODS[0]}" -c tensorfold -- python3 -c "import urllib.request;urllib.request.urlopen('http://localhost:8888/health',timeout=3)" >/dev/null 2>&1; then echo ready; exit 0; fi
      if "${K[@]}" exec "${PODS[0]}" -c tensorfold -- grep -qE "Traceback|FATAL|Error:" /cache/dev/log/rank0.log 2>/dev/null; then echo FAILED; exit 2; fi
      sleep 10
    done
    echo timeout; exit 3 ;;
  status)
    for p in "${PODS[@]}"; do "${K[@]}" exec "$p" -c tensorfold -- bash -c 'pgrep -fa "tensorfold serve|nsys" | cut -c1-100; grep MemAvailable /proc/meminfo' || true; done ;;
  log)
    r=${2:-0}; "${K[@]}" exec "${PODS[$r]}" -c tensorfold -- tail -n "${3:-40}" /cache/dev/log/rank$r.log ;;
esac
