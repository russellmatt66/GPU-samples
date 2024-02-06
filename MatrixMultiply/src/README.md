# Current Tasks
main.cu
- Understand why code can run with `N = 2^21`, should only be able to run up to `N = 2^14`.
- Add host code to validate against

CMakeLists.txt
- Implement 

# Directory Structure
main.cu (WIP)
- Currently implements square matrix multiplication
- Very much a demo. Initializes one matrix to be all zeros, and the other two to be random floats. 
- `$ nvcc -o matmul ../src/main.cu` (Run from inside build/)

CMakeLists.txt (WIP)
- Build file for CMake