#!/bin/bash

LOOPS=10

INPUT="/export/home1/ltarun/cuEFIM/datasets/connect_utility_spmf.txt"
MINUTIL=14500000
THREADS=16
TOTAL_MEMORY="370G"
MANAGED=true
OUTPUT="output.txt"
METHOD="DP"

for i in $(seq 1 $LOOPS)
do
    echo "Iteration $i"
    ./build/cuDPEFIM --input "$INPUT" --minutil "$MINUTIL" --threads "$THREADS" --total_memory "$TOTAL_MEMORY" --managed "$MANAGED" --output "$OUTPUT" --method "$METHOD"
done
