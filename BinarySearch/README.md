# Overview
Project to implement binary search with CUDA, and compare performance to CPU code.

GPU: GeForce GTX 960 (Maxwell 5.2)

# Current Tasks
(1) Test initialization functions from binarysearch.cu in testInitialize.cu

# Compile & Run Instructions
$nvcc -o binarysearch-binary binarysearch.cu
$./binarysearch-binary Ni Nx num_threads_per_block
