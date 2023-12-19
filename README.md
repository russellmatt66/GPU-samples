# Directory Structure
BinarySearch/ (FIN)
- CUDA implementation of binary search
- GPU: GeForce GTX 960 (Maxwell 5.2)
- Effective Bandwidth = (6 * log2(8192) * 4 * 2^27) / (941444937*10^-9) / 10^9 = 44.5 GB/s (estimate)
-- Ni = 2^27 particles (1 particle = 1 float)
-- Nx = 2^13 gridpoints (binary search gives log(Nx) * (Br + Bw) number of total reads and writes)
-- Br + Bw = 6 (Br = 5, Bw = 1)
- Kernel walltime = .941 [s] to find 2^27 particles in a grid of 2^13 gridpoints

InnerProduct/ (FIN)
- CUDA implementation of inner product

Transpose/ (WIP)
- CUDA implementation of matrix transpose 