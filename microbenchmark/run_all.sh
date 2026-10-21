#!/usr/bin/env bash
# Build and run the micro-benchmarks one at a time.
#   ./run_all.sh              all of 01..14
#   ./run_all.sh 03 07        only those
# Env: CUDA_VISIBLE_DEVICES (default 0), ARCH (default sm_90), LIMIT seconds per benchmark (default 60).
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ARCH="${ARCH:-sm_90}"
LIMIT="${LIMIT:-60}"
mkdir -p build results

if [ "$#" -gt 0 ]; then
  dirs=()
  for n in "$@"; do dirs+=("$(ls -d "${n}"_*/ | head -n1)"); done
else
  dirs=([0-9][0-9]_*/)
fi

for dir in "${dirs[@]}"; do
  name="${dir%/}"
  echo ">> $name"
  nvcc -O3 -std=c++17 -arch="$ARCH" -Icommon "$name/main.cu" -o "build/$name" -lcuda
  timeout -k 5 "$LIMIT" "build/$name" > "results/$name.csv"
  echo "   $(($(wc -l < "results/$name.csv") - 1)) rows -> results/$name.csv"
done
