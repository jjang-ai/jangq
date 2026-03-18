#!/bin/bash
# RAM monitor - logs every 5 seconds
# Usage: bash ram_monitor.sh &
LOG="/tmp/ram_monitor.log"
echo "$(date): RAM monitor started (PID $$)" > "$LOG"
while true; do
    ts=$(date +%H:%M:%S)
    # Count python processes and their RSS
    py_info=$(ps aux | grep -i python | grep -v grep | grep -v Chrome | awk '{sum+=$6; n++} END {printf "%d procs %dMB", n, sum/1024}')
    # Memory pressure
    free=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
    free_mb=$((free * 16384 / 1048576))
    echo "$ts | free=${free_mb}MB | $py_info" >> "$LOG"
    # Alert if free < 500MB
    if [ "$free_mb" -lt 500 ]; then
        echo "$ts | DANGER: free=${free_mb}MB | $py_info" | tee -a "$LOG"
    fi
    sleep 5
done
