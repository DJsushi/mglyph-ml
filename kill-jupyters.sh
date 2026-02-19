#!/bin/bash

# Kill all zombie/hung jupyter notebook kernels

echo "Killing jupyter notebook kernels..."

# Find and kill all ipykernel processes for the current user
pids=$(ps aux | grep "[i]pykernel" | awk '{print $2}')

if [ -z "$pids" ]; then
    echo "No jupyter kernel processes found."
    exit 0
fi

# Kill each process
kill_count=0
for pid in $pids; do
    echo "Killing PID $pid..."
    kill -9 "$pid" 2>/dev/null
    ((kill_count++))
done

echo "Killed $kill_count jupyter kernel process(es)."
