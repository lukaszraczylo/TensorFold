#!/usr/bin/env bash
# waits for the first queue, then sweeps the draft-stop threshold further
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
until grep -q "queue done" /tmp/glm-trials/queue.log 2>/dev/null; do sleep 20; done
run() { name=$1; shift; echo "=== $name $*" >> /tmp/glm-trials/queue.log; TRIAL_PRE="${PRE:-}" ./trial.sh "$name" "$@" 2>&1 | grep -E "^LEAN|ready|FAILED|timeout" >> /tmp/glm-trials/queue.log; }
PRE="" run t10 "TF_GLM_DFLASH_POLICY=fnc7:0.5"
PRE="" run t11 "TF_GLM_DFLASH_POLICY=fnc7:0.6"
echo "=== queue2 done" >> /tmp/glm-trials/queue.log
