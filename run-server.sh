#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

RUN_NR=$(ls .exp-count | wc -w)
echo "RUN_NR=$RUN_NR"
#echo $(($RUN_NR - 1))

python server.py --group_name="run-$RUN_NR"

touch .exp-count/run-$RUN_NR

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
