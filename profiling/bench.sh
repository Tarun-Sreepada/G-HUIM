#!/usr/bin/env bash
# Best-of-N mining kernel time (the "kernels" log line) per config.
# Usage: profiling/bench.sh <binary> <label> "<search:method ...>" [minutils] [repeats]
BIN=$1; LABEL=$2; CFGS=${3:-"bfs:static bfs:adaptive dfs:static dfs:adaptive"}
MINUTILS=${4:-"8000000 6000000"}; N=${5:-3}
D=${D:-/export/home1/ltarun/G-HUIM/datasets/accidents_utility_spmf.txt}
OUT=$(dirname "$0")/data/bench.csv
for M in $MINUTILS; do for c in $CFGS; do
  s=${c%%:*}; m=${c##*:}; best=; pat=
  for i in $(seq $N); do
    out=$(CUDA_VISIBLE_DEVICES=0 timeout -k 5 120 $BIN --input $D --minutil $M --search $s --method $m --total-memory 8G ${EXTRA:-} 2>&1)
    k=$(grep -oP 'kernels \K[\d.]+(?= s)' <<<"$out"); pat=$(grep -oP 'Number of patterns: \K\d+' <<<"$out")
    [[ -z $k ]] && { best=ERR; break; }
    best=$(python3 -c "print(min(float('$k'), float('${best:-9e9}')))")
  done
  echo "$LABEL,$s,$m,$M,$best,$pat" | tee -a $OUT
done; done
