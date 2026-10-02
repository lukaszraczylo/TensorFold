#!/usr/bin/env bash
# queue.sh : runs trials listed one per line in /tmp/glm-trials/queue.in as  name|ENV VARS|src-dir (src optional),
# appending LEAN results to /tmp/glm-trials/queue.log. A line "STOP" ends it. New lines can be appended while it runs.
D=$(cd "$(dirname "$0")" && pwd); cd "$D"; Q=/tmp/glm-trials/queue.in; L=/tmp/glm-trials/queue.log; n=0
BASE_SRC=/Users/nvm/Documents/projects/private/TensorFold/src
touch $Q
while true; do
  line=$(sed -n "$((n+1))p" $Q)
  if [ -z "$line" ]; then sleep 15; continue; fi
  n=$((n+1)); [ "$line" = STOP ] && { echo "=== queue stopped" >> $L; exit 0; }
  IFS='|' read -r name env src <<<"$line"; src=${src:-$BASE_SRC}
  echo "=== $name [$env] src=$src" >> $L
  ./trial.sh "$name" "$env" "$src" 2>&1 | grep -E "^LEAN|FAILED|timeout" >> $L
done
