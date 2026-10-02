#!/usr/bin/env bash
# run_trials.sh : the env-only trial queue; one line per trial in /tmp/glm-trials/queue.log
D=$(cd "$(dirname "$0")" && pwd); cd "$D"
run() { name=$1; shift; echo "=== $name $*" >> /tmp/glm-trials/queue.log; TRIAL_PRE="${PRE:-}" ./trial.sh "$name" "$@" 2>&1 | grep -E "^LEAN|ready|FAILED|timeout" >> /tmp/glm-trials/queue.log; }
mkdir -p /tmp/glm-trials
PRE="" run t04 "TF_GLM_DFLASH_POLICY=fnc7:0.25"
PRE="" run t05 "TF_GLM_DFLASH_POLICY=fnc7:0.4"
PRE="" run t06 "TF_GLM_EXL3_LOADS=nc2"
PRE="" run t07 "TF_GLM_EXL3_LOADS=nc4"
PRE="taskset -c 5-9,15-19" run t08 ""
PRE="" run t09 "TF_GLM_L2PF=0"
echo "=== queue done" >> /tmp/glm-trials/queue.log
