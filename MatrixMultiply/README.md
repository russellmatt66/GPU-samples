# Overview
Project to implement matrix multiplication with CUDA, benchmark the result, and teach a machine how to do it

# Current Tasks
(1) Benchmark `./build/matmul`
- Need to refactor `../src/matmul.cu` to separate the CPU code out
- CPU code is the rate limiting step, and necessitates multi-threading of the benchmarking
    - However, RTX2060 does not have the RAM, and benchmarking it doesn't need multi-threading, to run `matmul` in parallel (currently 8) past `N=256` 

# Project Status

# Directory Structure (WIP)
build/
-

data/
- 

debug/
- 

include/
- CUDA kernels moved to src

src/
- Currently implements square matrix multiplication. 

test/
-

machine-learning/
