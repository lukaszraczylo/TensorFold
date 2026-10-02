#!/usr/bin/env bash
# validates the draft-stop threshold winner: 4-stream aggregate for 0.3 and 0.4, exactness and a second lean run for 0.4
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; L=/tmp/glm-trials/queue.log; REF="$D/../../experiments/exact-ref.json"
boot() { export TRIAL_ENV="$1"; ./devctl.sh stop >/dev/null 2>&1; ./devctl.sh start >/dev/null; ./devctl.sh wait 1500 | tail -1 >> $L; }
echo "=== validate" >> $L
boot "TF_GLM_DFLASH_POLICY=fnc7:0.3"
python3 par_bench.py n4-fnc7-0.3 http://10.0.1.110:8888 /tmp/glm-trials/n4-0.3.json 4 5 2>&1 | grep SUMMARY >> $L
boot "TF_GLM_DFLASH_POLICY=fnc7:0.4"
python3 exact_check.py compare "$REF" 2>&1 | tail -1 >> $L
python3 par_bench.py n4-fnc7-0.4 http://10.0.1.110:8888 /tmp/glm-trials/n4-0.4.json 4 5 2>&1 | grep SUMMARY >> $L
python3 lean_bench.py t12-confirm-0.4 http://10.0.1.110:8888 /tmp/glm-trials/t12.json 2>&1 | grep -E "^LEAN" >> $L
echo "=== validate done" >> $L
