# Directory Structure
cuda_malloc.cu
- Debugging allocation
- Problem:`cudaMallocManaged` would create problems when trying to allocate matrices
- Solution: Use `cudaMalloc`