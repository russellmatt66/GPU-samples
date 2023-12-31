# Overview
Project to implement binary search with CUDA, and then find the configuration with the best performance by performing a parameter sweep of the kernel. 

# Effective Bandwidth Calculation
GPU: GeForce GTX 960 (Maxwell 5.2)
Effective Bandwidth = (4 * log2(8192) * 4 * 2^27) / (941444937*10^-9) / 10^9 ~ 30 GB/s (estimate)
- Ni = 2^27 particles (1 particle = 1 float)
- Nx = 2^13 gridpoints (binary search gives log(Nx) * (Br + Bw) number of total reads and writes)
- Br + Bw = 4 (see explanation in 'machine-learning/analyze.py')

Kernel walltime = .941 [s] to find 2^27 particles in a grid of 2^13 gridpoints

# Current Tasks
(1) Write Python 
- To clean kernel benchmarking data
- To analyze kernel benchmarking data
- For large data volumes, the output from the CUDA timer library is incoherent, therefore implement an ML model using sklearn in order to predict the execution configuration performance
(2) Run kernel parameter sweep on RTX 2060


# Directory Structure
binarysearch.cu
- Code to benchmark binary search CUDA kernel
- **Appends** data to .csv files

automate-benchmarking.sh
- Shell script that automates the benchmarking of the binary search CUDA kernel
- Need to run `find ./benchmarking-data/ -type f ! -name 'README.md' -exec rm -rf {} +` beforehand to delete everything 

benchmarking-data/
- Storage for benchmarking data

machine-learning/
- Contains Python code to analyze the performance data and predict the performance as a function of execution configuration 

binarysearch-validate.cu
- Code to validate binary search using linear search
- Linear search kernel will time out before GTX 960 VRAM fills up, only have a single GPU in machine so it's doing both display and compute, therefore watchdog timer

# Compile & Run Instructions
$nvcc -o benchmark binarysearch.cu

**BEFORE RUNNING, MAKE SURE TO DELETE ANY PREVIOUS DATA IN BENCHMARKING-DATA**
$./benchmark Ni Nx SM_multiplier num_threads_per_block nruns
- Ni: log2(number of particles)
- Nx: log2(number of gridpoints)
- SM_multiplier: determines the number of blocks in the execution configuration (=SM_multiplier * numberOfSMs)
- num_threads_per_block: the number of threads per block in the execution configuration
- nruns: the number of iterations to run a given configuration for
