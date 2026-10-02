#!/usr/bin/env bash
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; L=/tmp/glm-trials/queue.log; SRC=/Users/nvm/Documents/projects/private/TensorFold/src
run() { name=$1; shift; echo "=== $name $*" >> $L; ./trial.sh "$name" "$@" 2>&1 | grep -E "^LEAN|FAILED|timeout" >> $L; }
run t14-async "TF_GLM_MULTI_ASYNC=1" $SRC
run t15-packed "TF_GLM_MULTI_SAMPLER=packed" $SRC
run t16-both "TF_GLM_MULTI_ASYNC=1 TF_GLM_MULTI_SAMPLER=packed" $SRC
echo "=== queue3 done" >> $L
