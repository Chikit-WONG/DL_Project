#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_FILE="${1:-$ROOT/plan/full_training_set_rerun_jobs_latest.txt}"

if [ ! -f "$JOB_FILE" ]; then
  echo "Missing job file: $JOB_FILE" >&2
  exit 1
fi

job_ids=$(grep -E '^[a-zA-Z0-9_]+=' "$JOB_FILE" | cut -d= -f2 | paste -sd, -)

if [ -z "$job_ids" ]; then
  echo "No job ids found in $JOB_FILE" >&2
  exit 1
fi

echo "=== squeue ==="
squeue -j "$job_ids" -o '%.10i %.9P %.20j %.8T %.10M %.6D %R'
echo
echo "=== sacct ==="
sacct -j "$job_ids" --format=JobID,JobName%28,Partition,State,Elapsed,ExitCode -n -P
