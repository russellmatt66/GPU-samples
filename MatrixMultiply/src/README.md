# Current Tasks
main.cu
- [SOLVED]Understand why code can run with `N = 2^21`, should only be able to run up to `N = 2^14`.
    - Problem with `cudaMallocManaged`, use `cudaMalloc` instead.
- [ADDED]Add host code to validate against

CMakeLists.txt [IMPLEMENTED]
- Implement 

# Directory Structure
matmul.cu (FIN)
- Currently implements square matrix multiplication
- Initializes one matrix to be all zeros, and the other two to be random floats. 
- `$ nvcc -o matmul ../src/main.cu` (Run from inside build/)

CMakeLists.txt (FIN)
- Build file for `cmake`