#include "binarytree.c"

__global__ void simulateSearch(int* sum, const int* num_iters, const int* p_cells, const int N){
    // num_iters 
    // - integer array whose values correspond to how many iterations it takes to find a particle in that cell
    // - size: Nx
    // - the data for the values comes from a binary tree that represents the various outcomes
    // p_cells 
    // - integer array that corresponds to the grid-cells where the individuals of a population of particles can be found
    // - size: N
    // - p_cells[i] \in [0,Nx-1]
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    int partial = 0;

    for (int i = tidx; i < N; i+=nthreads){
        partial += num_iters[p_cells[i]]; // tidy
    }

    atomicAdd(sum, partial);
}