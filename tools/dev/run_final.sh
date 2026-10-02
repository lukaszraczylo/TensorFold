#!/usr/bin/env bash
# final confirmation: A = segment-4 baseline L2 prefetch (auto blocks, 8 MB), B = the stack (blocks 8, 16 MB, 4 KB pieces)
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; L=/tmp/glm-trials/final.log; SRC=/Users/nvm/Documents/projects/private/TensorFold/src; REF="$D/../../experiments/exact-ref.json"
echo "=== final" > $L
./trial.sh fin-A "TF_GLM_L2PF_BLOCKS=0 TF_GLM_L2PF_MB=8" $SRC 2>&1 | grep -E "^LEAN|FAILED|timeout" >> $L
./trial.sh fin-B "TF_GLM_L2PF_KB=4" $SRC 2>&1 | grep -E "^LEAN|FAILED|timeout" >> $L
python3 exact_check.py compare "$REF" 2>&1 | tail -1 >> $L
python3 par_bench.py n4-fin-B http://10.0.1.110:8888 /tmp/glm-trials/n4-fin-B.json 4 5 2>&1 | grep SUMMARY >> $L
./trial.sh fin-A2 "TF_GLM_L2PF_BLOCKS=0 TF_GLM_L2PF_MB=8" $SRC 2>&1 | grep -E "^LEAN|FAILED|timeout" >> $L
python3 par_bench.py n4-fin-A2 http://10.0.1.110:8888 /tmp/glm-trials/n4-fin-A2.json 4 5 2>&1 | grep SUMMARY >> $L
echo "=== final done" >> $L
