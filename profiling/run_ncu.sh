#!/usr/bin/env bash
# Per-kernel hardware counters for the heaviest launches of each config (needs GPU counter access).
#   Run once as root:   sudo bash profiling/run_ncu.sh
# or enable counters for everyone (admin, driver reload needed):
#   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf
# Safety: one GPU, minutil 8M (0.5 s of mining), a few launches per kernel family, every run under timeout.
set -u
cd "$(dirname "$0")/results" || exit 1
NCU=${NCU:-/usr/local/cuda-12.8/bin/ncu}
BIN=$PWD/../../build/G-HUIM
D=${D:-/export/home1/ltarun/G-HUIM/datasets/accidents_utility_spmf.txt}
M=8000000
SECTIONS=(SpeedOfLight MemoryWorkloadAnalysis MemoryWorkloadAnalysis_Chart MemoryWorkloadAnalysis_Tables ComputeWorkloadAnalysis
          Occupancy LaunchStats WarpStateStats SchedulerStats InstructionStats SourceCounters)
sec=(); for s in "${SECTIONS[@]}"; do sec+=(--section "$s"); done
# Extra raw metrics for the layout question: coalescing, cache hit rates, atomics, DRAM traffic.
METRICS=l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum,\
dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_membar_per_warp_active.pct,smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct,\
smsp__warps_active.avg.pct_of_peak_sustained_active,smsp__issue_active.avg.pct_of_peak_sustained_active
run() {  # name search method kernel-regex skip count
  local name=$1 s=$2 m=$3 re=$4 skip=$5 cnt=$6
  CUDA_VISIBLE_DEVICES=0 timeout -k 10 900 "$NCU" "${sec[@]}" --metrics "$METRICS" --import-source yes \
    --cache-control all --clock-control base -k "regex:$re" -s "$skip" -c "$cnt" -f -o "ncu_${name}" \
    "$BIN" --input "$D" --minutil $M --search $s --method $m --total-memory 8G > "ncu_${name}.log" 2>&1
  echo "ncu_${name} rc=$?"
}
# Early launches (levels/batches 1-3) hold most of the time; the later skip lands on the many-task levels.
run bfs_static_block   bfs static   'block_tasks'                       0 3
run bfs_static_block_late bfs static 'block_tasks'                      8 2
run bfs_adaptive_block bfs adaptive 'block_tasks'                       0 3
run bfs_adaptive_warp  bfs adaptive 'warp_tasks'                        0 3
run bfs_adaptive_sub   bfs adaptive 'subwarp_tasks'                     20 3
run bfs_adaptive_huge  bfs adaptive 'huge_(local|subtree|merge|write)'  0 4
run dfs_static_block   dfs static   'block_tasks'                       0 3
run dfs_static_block_late dfs static 'block_tasks'                      8 2
run dfs_adaptive_block dfs adaptive 'opt::.*block_tasks'                0 3
run dfs_adaptive_warp  dfs adaptive 'opt::.*warp_tasks'                 0 3
run dfs_adaptive_sub   dfs adaptive 'opt::.*subwarp_tasks'              20 3
run dfs_adaptive_huge  dfs adaptive 'huge_(local|subtree|merge|write)'  0 4
run dfs_select         dfs adaptive 'select_kernel'                     0 3
echo finished
