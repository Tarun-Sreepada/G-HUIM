#!/usr/bin/env bash
# Run gate: every supported search/method pair must equal EFIM's pattern count, with default and small memory.
# Usage: profiling/gate.sh [methods...]   (default: static adaptive warp)
BIN=${BIN:-build/G-HUIM}
D=${D:-/export/home1/ltarun/G-HUIM/datasets/accidents_utility_spmf.txt}
METHODS=("$@"); [ ${#METHODS[@]} -eq 0 ] && METHODS=(static adaptive warp)
declare -A REF=([14000000]=143536 [8000000]=3030087 [6000000]=10079776)
fail=0
for mem in default 4G; do
  for M in 14000000 8000000 6000000; do
    for s in bfs dfs; do for m in "${METHODS[@]}"; do
      [[ $s == dfs && $m == warp ]] && continue   # BFS-only for now
      args=(--input "$D" --minutil $M --search $s --method $m)
      [ $mem != default ] && args+=(--total-memory $mem)
      out=$(CUDA_VISIBLE_DEVICES=0 timeout -k 5 120 $BIN "${args[@]}" 2>&1); rc=$?
      n=$(grep -oP 'Number of patterns: \K\d+' <<<"$out")
      k=$(grep -oP 'kernels \K[\d.]+(?= s)' <<<"$out")
      st=OK; [ "$n" != "${REF[$M]}" ] && { st="FAIL(rc=$rc got=$n want=${REF[$M]})"; fail=1; }
      echo "mem=$mem minutil=$M $s-$m patterns=$n kernels=${k}s $st"
      [[ $st != OK ]] && echo "$out" | tail -3
    done; done
  done
done
exit $fail
