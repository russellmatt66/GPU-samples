# Overview
Project to implement binary search with CUDA, and compare performance to CPU code.

GPU: GeForce GTX 960 (Maxwell 5.2)

# Current Tasks
(1) Refactor to separate linear and binary search 

# Directory Structure
binarysearch-validate.cu
- Code to validate binary search using linear search
- Linear search kernel times out, only have a single GPU in machine so it's doing both display and compute, therefore watchdog timer

# Compile & Run Instructions
(replace binarysearchvalidate-binary with a better, more concise name)
$nvcc -o binarysearchvalidate-binary binarysearch-validate.cu
$./binarysearchvalidate-binary Ni Nx num_threads_per_block
