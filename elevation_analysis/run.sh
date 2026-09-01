#!/usr/bin/env bash
set -euo pipefail

# Edit parameters here; run this script with no command-line arguments.
INPUT_CSV="splitData/robotstate_40000_env16_run16.csv"
RUN_ALL=false # Set true to process every CSV in splitData.
OUTPUT_ROOT="global_terrain"
GLOBAL_RESOLUTION_M="0.01"
LOCAL_GRID_RESOLUTION_M="0.04"
FLIP_ROW=false  # Set true if increasing CSV rows mean local -y.
FLIP_COL=false  # Set true if increasing CSV columns mean local -x.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_one() {
  local input_csv="$1"
  local csv_name output_dir
  csv_name="$(basename "$input_csv" .csv)"
  output_dir="$OUTPUT_ROOT/$csv_name"
  converter_args=(
    "$input_csv"
    --output-dir "$output_dir"
    --global-resolution "$GLOBAL_RESOLUTION_M"
    --local-resolution "$LOCAL_GRID_RESOLUTION_M"
  )
  [[ "$FLIP_ROW" == true ]] && converter_args+=(--flip-row)
  [[ "$FLIP_COL" == true ]] && converter_args+=(--flip-col)

  MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-codex" \
    python3 build_global_terrain.py "${converter_args[@]}"
}

if [[ "$RUN_ALL" == true ]]; then
  shopt -s nullglob
  csv_files=(splitData/*.csv)
  (( ${#csv_files[@]} > 0 )) || { echo "No CSV files found in splitData" >&2; exit 1; }
  for input_csv in "${csv_files[@]}"; do
    run_one "$input_csv"
  done
else
  run_one "$INPUT_CSV"
fi
