#!/usr/bin/env bash
# Builds efim with link-time optimisation and profile-guided optimisation.
#   efim/pgo.sh <dataset> [build-dir]     (default build dir: build-pgo)
# Phase 1 builds an instrumented binary, phase 2 runs it on training workloads (single- and
# multi-threaded, several minutil values, with output), phase 3 rebuilds using the profile.
# The result is <build-dir>/efim. Re-run after changing the hot code.
set -euo pipefail
dataset=${1:?usage: efim/pgo.sh <dataset> [build-dir]}
root=$(cd "$(dirname "$0")/.." && pwd)
build=${2:-$root/build-pgo}
profile=$build/efim-profile
rm -rf "$profile"

cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DEFIM_PGO=generate -DEFIM_PGO_DIR="$profile" >/dev/null
cmake --build "$build" --target efim -j"$(nproc)"

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
head -n 60000 "$dataset" > "$out/slice.txt"
run() { timeout 900 "$build/efim" --separator space --log-level warn "$@" >/dev/null; }
# Minutil values scale with the dataset; these suit accidents_utility_spmf.txt.
run -i "$out/slice.txt" -m 2000000 -o "$out/o1.txt" -t 1
run -i "$out/slice.txt" -m 1200000 -o "$out/o2.txt" -t 8
run -i "$dataset" -m 12000000 -o "$out/o3.txt" -t 1
run -i "$dataset" -m 8000000 -o "$out/o4.txt" -t 48
run -i "$dataset" -m 5000000 -t 48

cmake -S "$root" -B "$build" -DEFIM_PGO=use >/dev/null
cmake --build "$build" --target efim -j"$(nproc)"
echo "built $build/efim"
