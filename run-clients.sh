#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

RUN_NR=$(ls .exp-count | wc -w)
echo "RUN_NR=$RUN_NR"

for i in $(seq 0 $(($1 - 1))); do
    python client.py --group_name="run-$RUN_NR" --node-id="$i" --client_count="$1" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
