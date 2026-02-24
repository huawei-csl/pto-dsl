#!/bin/bash

# =========
# Usage: `bash ./repeat_runs.sh`
# To find non-empty error logs: `find logs/ -name "err_*.log" -not -size 0`
# ==========

# Configuration
total_runs=100
log_dir="logs"
success_count=0
failure_count=0

# Create the log directory
mkdir -p "$log_dir"

echo "Starting $total_runs iterations. Logging to ./$log_dir..."
echo "-------------------------------------------------------"

for ((i=1; i<=total_runs; i++))
do
    # Format the index with leading zeros (e.g., 001, 002)
    suffix=$(printf "%03d" $i)
    
    # Define log paths
    out_file="$log_dir/out_$suffix.log"
    err_file="$log_dir/err_$suffix.log"

    # Execute the command
    # Standard output to out_file, Standard error to err_file
    python ./run_add.py > "$out_file" 2> "$err_file"
    
    # Update counts based on exit status
    if [ $? -eq 0 ]; then
        ((success_count++))
    else
        ((failure_count++))
    fi

    # Progress indicator: Report every 2 runs
    if (( i % 2 == 0 )); then
        echo "Progress: $i/$total_runs | Successes: $success_count | Failures: $failure_count"
    fi
done

echo "-------------------------------------------------------"
echo "Final Summary:"
echo "Total Runs: $total_runs"
echo "Successes:  $success_count"
echo "Failures:   $failure_count"
echo "-------------------------------------------------------"