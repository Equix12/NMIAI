#!/bin/bash
# Watch for new active round and run solver
# Also updates calibration data after each round completes
# Usage: bash watch_and_solve.sh

cd /home/haava/NMAI
VENV=".venv/bin/python"

echo "Watching for new active round..."
LAST_ROUND=0

while true; do
    STATUS=$($VENV -c "
import requests
from pathlib import Path
with open('.env') as f:
    token = next(l.strip().split('=',1)[1] for l in f if l.startswith('NMAI_TOKEN='))
s = requests.Session()
s.headers['Authorization'] = f'Bearer {token}'
rounds = s.get('https://api.ainm.no/astar-island/rounds').json()
active = [r for r in rounds if r['status'] == 'active']
if active:
    budget = s.get('https://api.ainm.no/astar-island/budget').json()
    if budget['queries_used'] == 0:
        print(f'NEW_ROUND {active[0][\"round_number\"]}')
    else:
        print(f'ACTIVE_USED {active[0][\"round_number\"]} {budget[\"queries_used\"]}/{budget[\"queries_max\"]}')
else:
    print('NO_ACTIVE')
" 2>/dev/null)

    if [[ $STATUS == NEW_ROUND* ]]; then
        ROUND=$(echo $STATUS | awk '{print $2}')

        # Update calibrations with any newly completed rounds first
        echo "$(date): Updating calibrations..."
        $VENV astar-island/update_calibrations.py 2>&1

        echo "$(date): New round $ROUND detected! Running solver..."
        $VENV astar-island/solve_v3.py 2>&1 | tee astar-island/solve_round${ROUND}.log
        LAST_ROUND=$ROUND
        echo "$(date): Solver complete for round $ROUND."
    elif [[ $STATUS == ACTIVE_USED* ]]; then
        # Only print occasionally to avoid spam
        ROUND=$(echo $STATUS | awk '{print $2}')
        if [[ $ROUND != $LAST_ROUND ]]; then
            echo "$(date): $STATUS (already in progress)"
            LAST_ROUND=$ROUND
        fi
    else
        echo -n "."
    fi
    sleep 60
done
