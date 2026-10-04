#!/usr/bin/env bash
# Configuration: SECONDS_LIMIT, THREADS, SEED, RUN_DIR, PYTHON, MEMORY_GB.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-.venv/bin/python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Run bash scripts/setup_linux.sh first, or set PYTHON to your environment's Python." >&2
    exit 3
fi
seconds_limit="${SECONDS_LIMIT:-43200}"
threads="${THREADS:-$("$python_bin" -c 'import os; print(min(16, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)))')}"
seed="${SEED:-0}"
run_dir="${RUN_DIR:-runs/$(date -u +%Y%m%dT%H%M%SZ)-seed${seed}}"
# Refuse to overwrite an earlier experiment's checkpoint or logs.
if [[ -d "$run_dir" ]] && [[ -n "$(ls -A -- "$run_dir")" ]]; then
    echo "RUN_DIR already contains files: $run_dir. Choose a new directory." >&2
    exit 3
fi
mkdir -p -- "$run_dir/nodes"
run_dir="$(cd -- "$run_dir" && pwd)"
extra_params=(--gurobi-param NodefileStart=0.5)
if [[ -n "${MEMORY_GB:-}" ]]; then
    extra_params+=(--gurobi-param "SoftMemLimit=${MEMORY_GB}")
fi
echo "12-hour profile: seconds=$seconds_limit threads=$threads seed=$seed output=$run_dir"
echo "No positional weights or clue-spread requirements. Known 17-clue lower bound enabled."
set +e
"$python_bin" -u python/sudoku_benders_gurobi.py \
    --size 9 --mode callback --cut unavoidable --separator hybrid --batch-size 8 \
    --time-limit "$seconds_limit" --threads "$threads" --subproblem-threads 1 \
    --strengthen --row-symmetry --min-clues 17 --warm-start --seed "$seed" \
    --gurobi-param MIPFocus=1 \
    --gurobi-param "NodefileDir=$run_dir/nodes" \
    --gurobi-log --run-dir "$run_dir" --output-json "$run_dir/result.json" \
    "${extra_params[@]}" "$@" 2>&1 | tee "$run_dir/run.log"
statuses=("${PIPESTATUS[@]}")
set -e
if [[ "${statuses[1]}" -ne 0 ]]; then
    echo "Cannot write run.log (tee exit ${statuses[1]})." >&2
    exit "${statuses[1]}"
fi
status="${statuses[0]}"
if [[ "$status" -eq 2 ]] && [[ -f "$run_dir/result.json" ]]; then
    # Time/memory limits are expected experiment outcomes, not launcher errors.
    "$python_bin" -c 'import json, sys; r=json.load(open(sys.argv[1])); print("Finished:", r["status"], "best clues:", r["clue_count"], "unique:", r["verified_unique"]); sys.exit(0 if r["status"] in ("time_limit", "gurobi_status_17", "verification_limit", "master_iteration_limit") else 2)' "$run_dir/result.json"
else
    exit "$status"
fi
