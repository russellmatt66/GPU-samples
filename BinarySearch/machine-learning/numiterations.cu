#include <stdio.h>
#include <cstdlib>
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#include <fstream>

#include "binarytree.h"

/*
Code to produce binary for usage in clean.py, so that accurate values of the effective bandwidth can be determined 
*/

// Binary tree functionality for device 
struct d_BTNode {
    int cell;
    int num_iter;
    d_BTNode* left;
    d_BTNode* right;
};

__device__ d_BTNode* d_createBTNode(int cell, int num_iter){
    d_BTNode* d_newNode = (d_BTNode*)(sizeof(d_BTNode));
    if (d_newNode != NULL){
        d_newNode->cell = cell;
        d_newNode->num_iter = num_iter;
        d_newNode->left = NULL;
        d_newNode->right = NULL;
    }
    return d_newNode; 
}

// Build the binary tree
__global__ void buildLeaves(d_BTNode* parent, int Nx, int low, int high, int guess, int level){
    if (parent == NULL || level > (int)log2(Nx)){
        return;
    }
    int left_low = low; 
    int left_high = guess;
    int left_guess = (left_low + left_high) / 2;
    int right_low = guess;
    int right_high = high;
    int right_guess = (right_low + right_high) / 2;
    d_BTNode* leftNode = d_createBTNode(left_guess, level);
    d_BTNode* rightNode = d_createBTNode(right_guess, level);
    parent->left = leftNode;
    parent->right = rightNode;
    buildLeaves(parent->left, Nx, left_low, left_high, left_guess, level + 1);
    buildLeaves(parent->right, Nx, right_low, right_high, right_guess, level + 1);
}

// 
void writeBST(BTNode* root, int jump, int *counter, std::ofstream& bst_file){
    if (root == NULL){
        return;
    }
    else if ((*counter) % jump == 0){
        bst_file << "Node " << *counter << " represents looking in cell " << root->val << ", where it would take " 
            << root->depth << " iterations to find a particle there" << std::endl; 
    }
    (*counter)++;
    writeBST(root->left, jump, counter, bst_file);
    (*counter)++;
    writeBST(root->right, jump, counter, bst_file);
}

__device__ void getNumIterations(d_BTNode*, int, const int);

// Function to simulate how many total iterations binary search takes to find a given population of particles 
__global__ void simulateSearch(int* sum, const int* num_iters, const int* p_cells, const int N){
    // num_iters 
    // - integer array whose values correspond to how many iterations it takes to find a particle in that cell
    // - size: Nx - 1
    // - the data for the values comes from a binary tree that represents the various outcomes
    // p_cells 
    // - integer array that corresponds to the grid-cells where the individuals of a population of particles can be found
    // - size: N
    // - p_cells[i] \in [0,Nx-2] (there are Nx-1 grid-cells)
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    int partial = 0;

    for (int i = tidx; i < N; i+=nthreads){
        partial += num_iters[p_cells[i]]; // tidy
    }

    atomicAdd(sum, partial);
}

__global__ void initializeNumIters(d_BTNode* root, int* num_iters, const int Nxm1){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for (int j = tidx; tidx < Nxm1; tidx += nthreads){
        getNumIterations(root, num_iters[j], j);
    }
}

__device__ void getNumIterations(d_BTNode* root, int curr_num_iter, const int j){
    // Depth-first search 
    if (root->cell == j){
        curr_num_iter = root->num_iter;
        return;
    }

    if (root->left != NULL){
        getNumIterations(root->left, curr_num_iter, j);
    }
    if (root->right != NULL){
        getNumIterations(root->right, curr_num_iter, j);
    }
    return;
}


__global__ void initializePCells(int* p_cells, const int N, const int Nx){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x; 
    int nthreads = blockDim.x * gridDim.x;

    unsigned long long seed = 1234;

    // Implement device random number generation with curand 
    curandState_t state;
    if (tidx < N){
        curand_init(seed, tidx, 0, &state);
    }

    // Get random integer between [0,Nx-2] to represent which cell the particle is in
    float aRandomValue = 0;
    for (int i = tidx; i < N; i += nthreads){
        aRandomValue = static_cast<float>(curand_uniform(&state)) * (Nx-2); 
        p_cells[i] = (int)aRandomValue;
    }
}

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[]){
    int N = std::stoi(argv[1]);
    int Nx = std::stoi(argv[2]);

    // Initialize device specific parameters
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // Create device data
    int *num_iters, *p_cells, *total_iters = 0;

    checkCuda(cudaMallocManaged(&num_iters, (Nx-1)*sizeof(int)));
    checkCuda(cudaMallocManaged(&p_cells, N*sizeof(int)));
    checkCuda(cudaMallocManaged(&total_iters, sizeof(int)));

    // Create binary tree with Nx nodes, representing binary search outcomes
    int low = 0, high = Nx-1;
    int guess = (low + high) / 2;
    int level = 1;
    BTNode* root = createBTNode(guess, level);
    buildLeaves(root, Nx, low, high, guess, level); // probably faster to implement a device version of this

    std::ofstream bst_file; 
    bst_file.open("bst.txt");
    int *counter = 0;
    int jump = Nx / 16;
    writeBST(root, jump, counter, bst_file);
    bst_file.close();



    // Define execution configuration
    int num_blocks = numberOfSMs;
    int num_threads_per_block = 32;

    // Initialize num_iters, and p_cells
    initializeNumIters<<<num_blocks, num_threads_per_block>>>(root, num_iters, Nx-1); // There are Nx-1 cells 
    initializePCells<<<num_blocks, num_threads_per_block>>>(p_cells, N, Nx);
    checkCuda(cudaDeviceSynchronize());

    // SANITY CHECK - Check the values of binary tree, num_iters, and p_cells
    std::ofstream p_cell_file, num_iter_file;

    p_cell_file.open("p_cells.txt");
    jump = N / 16;
    for (int i = 0; i < N; i += jump){
        p_cell_file << "Particle " << i << " is in cell " << p_cells[i] << std::endl;
    }
    p_cell_file.close();

    num_iter_file.open("num_iters.txt");
    jump = Nx / 16;
    for (int j = 0; j < Nx; j += jump){
        num_iter_file << "It takes " << num_iters[j] << " iterations to find a particle in cell " << j << std::endl;
    }
    num_iter_file.close();


    
    // Call CUDA kernels to simulate the binary search algorithm, and compute total number of iterations required  
    simulateSearch<<<num_blocks, num_threads_per_block>>>(total_iters, num_iters, p_cells, N);
    checkCuda(cudaDeviceSynchronize());

    // DO NOT DELETE, clean.py catches this value!
    printf("%d\n", *total_iters / N);

    // Free unified memory
    cudaFree(p_cells);
    cudaFree(num_iters);
    cudaFree(root);
    cudaFree(total_iters);
    return 0;
}