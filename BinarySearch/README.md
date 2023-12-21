# Overview
Project to implement binary search with CUDA, and then find the configuration with the best performance by performing a parameter sweep of the kernel. 

GPU: GeForce GTX 960 (Maxwell 5.2)

Effective Bandwidth = (6 * log2(8192) * 4 * 2^27) / (941444937*10^-9) / 10^9 = 44.5 GB/s (estimate)
- Ni = 2^27 particles (1 particle = 1 float)
- Nx = 2^13 gridpoints (binary search gives log(Nx) * (Br + Bw) number of total reads and writes)
- Br + Bw = 6 (Br = 5, Bw = 1)

Kernel walltime = .941 [s] to find 2^27 particles in a grid of 2^13 gridpoints

# Current Tasks
(1) Perform parameter sweep of binary search kernel

# Directory Structure
binarysearch-validate.cu
- Code to validate binary search using linear search
- Linear search kernel will time out before GTX 960 VRAM fills up, only have a single GPU in machine so it's doing both display and compute, therefore watchdog timer

# Compile & Run Instructions
(replace binarysearchvalidate-binary with a better, more concise name)
$nvcc -o benchmark binarysearch.cu
$./benchmark Ni Nx SM_multiplier num_threads_per_block nruns
- Ni: log2(number of particles)
- Nx: log2(number of gridpoints)
- SM_multiplier: determines the number of blocks in the execution configuration (=SM_multiplier * numberOfSMs)
- num_threads_per_block: the number of threads per block in the execution configuration
- nruns: the number of iterations to run a given configuration for
