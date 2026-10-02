#!/usr/bin/env bash
# nsys timelines: 4 search/method pairs x 2 workloads, sequential, one GPU, each under timeout.
cd "$(dirname "$0")/results" || exit 1
BIN=../../build/G-HUIM
D=${D:-/export/home1/ltarun/G-HUIM/datasets/accidents_utility_spmf.txt}
for M in 8000000 6000000; do
  for s in bfs dfs; do for m in static adaptive; do
    name=${s}_${m}_${M}
    CUDA_VISIBLE_DEVICES=0 timeout -k 10 300 nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
      --sample=none --cpuctxsw=none -f true -o $name \
      $BIN --input $D --minutil $M --search $s --method $m --total-memory 8G > $name.log 2>&1
    echo "$name rc=$? $(grep -oP 'Number of patterns: \K\d+' $name.log)"
    nsys export --type sqlite -f true -o $name.sqlite $name.nsys-rep > /dev/null 2>&1
  done; done
done
echo finished
