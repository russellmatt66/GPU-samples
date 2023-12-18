#define USE_MATH_DEFINES

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <fstream>


// Locates all the items (particles) in *item_positions, using binary search.
__global__ void BinarySearchGPU(const float *grid, const int Nx, const float *item_positions, const int Ni, int *item_indices){
    // Grid-Stride through array of item positions and call bs() for each element
    int tnum = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for (int i = tnum; i < Ni; i += nthreads){
        BinarySearchDevice(grid, Nx, item_positions[i], item_indices, i);
    }
}

// Implementation of Binary Search for use on the device
/* Do __device__ specified functions need to return void? */
__device__ void BinarySearchDevice(const float *grid, const int Nx, const float item_position, int *item_indices, const int i){
    /*
    N - size of grid 
    item_position - exactly where the object is located in the grid
    item_indices - the cells where the objects are in the grid
    i - the location in item_indices where the found index (j) should be placed
    */
    int low = 0;
    int high = Nx-1;
    int j = 0, counter = 0;

    while (low <= high){
        j = floor((low + high) / 2);
        counter++;
        if (grid[j] <= item_position && grid[j+1] > item_position){ // inside cell j
            item_indices[i] = j;
        }
        else if (item_position > grid[j]){ // item is to the right of cell j
            low = j+1;
        }
        else if (item_position < grid[j]){ // item is to the left of cell j
            high = j;
        }
        else if (counter >= sqrtf32(Nx)){ // It's not in the grid
            item_indices[i] = -1; 
        }
    }
}

/* Locates all the particles in *item_positions, using Linear Search */
__global__ void LinearSearchGPU(){
    /* Code goes here */
}

/* Device implementation of Linear Search */
__device__ void LinearSearchDevice(){
    /* Code goes here */
}

// Grid-Stride through grid and set uniformly spaced values
__global__ void InitializeGrid(float *grid, const float x_min, const float dx, const int Nx){
    int tnum = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for (int j = tnum; j < Nx; j += nthreads){
        grid[j] = x_min + j * dx;
    }
}

// Grid-Stride through grid and set random values for particles
/* Slower than it should bc of overhead from RNG */ 
__global__ void InitializeParticles(float *particle_positions, const int Ni, const unsigned long long seed, const float x_min, const float x_max){
    int tnum = threadIdx.x + blockDim.x * blockIdx.x; // tid or tnum is a better name for idiom
    int nthreads = blockDim.x * gridDim.x;

    // Implement device random number generation with curand 
    curandState_t state;
    if (tnum < Ni){
        curand_init(seed, tnum, 0, &state);
    }

    float aRandomValue = static_cast<float>(curand_uniform(&state));

    // Linear mapping from [0.0, 1.0] to [x_min, x_max]
    float b = x_min;
    float m = x_max - b;

    aRandomValue = m * aRandomValue + b; 

    for (int it = tnum; it < Ni; it += nthreads){
        particle_positions[it] = aRandomValue;
    }
}

// Just set all to -1
__global__ void InitializeIndices(int *item_indices, const int Ni){
    int tnum = threadIdx.x + blockDim.x * blockIdx.x; // tid or tnum is a better name for idiom
    int nthreads = blockDim.x * gridDim.x;
    
    for (int i = tnum; i < Ni; i += nthreads){
        item_indices[i] = -1;
    }
}

// CPU wrapper for locating all the objects that are within the grid, using binary search. 
void BinarySearchCPU(const float *grid, const int Nx, const float *item_positions, const int Ni, int *item_indices){
    for (int i = 0; i < Ni; i++){
        item_indices[i] = BinarySearchHost(grid, Nx, item_positions[i]);
    }
}

// Host implementation of binary search
int BinarySearchHost(const float *grid, const int Nx, const float item_position){
    int low = 0;
    int high = Nx-1;
    int j = 0, counter = 0;

    while (low <= high){
        j = floor((low + high) / 2);
        counter++;
        if (grid[j] <= item_position && grid[j+1] > item_position){ // inside cell j
            return j;
        }
        else if (item_position > grid[j]){ // item is to the right of cell j
            low = j+1;
        }
        else if (item_position < grid[j]){ // item is to the left of cell j
            high = j;
        }
        else if (counter >= sqrt(Nx)){ // It's not in the grid
            return -1; 
        }
    }

    return -1; // Not found
}

// Error-checking Macro from NVIDIA DLI:GSAC CUDA C/C++ course
inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}


// Driver Code
int main(int argc, char* argv[]){
    /* Initialize grid, and particles inside grid */
    int Ni = std::stoi(argv[1]); // Number of items
    int Nx = std::stoi(argv[2]); // Number of grid points

    // Device data
    float *x_grid, *particle_positions;
    int *item_indices;

    checkCuda(cudaMallocManaged(&x_grid, Nx));
    checkCuda(cudaMallocManaged(&particle_positions, Ni));
    checkCuda(cudaMallocManaged(&item_indices, Ni));

    float x_min = -M_PI, x_max = M_PI; /* Should I refactor this to accept these as input? */
    float dx = (x_max - x_min) / (float(Nx) - 1.0);

    // Set execution configuration 
    int num_blocks = 8; // GeForce GTX 960 has 8 SMs
    int num_threads_per_block = std::stoi(argv[3]);

    unsigned long long seed = 1234;

    // Call CUDA kernels to initialize device objects
    InitializeGrid<<<num_blocks, num_threads_per_block>>>(x_grid, x_min, dx, Nx); // uniformly-spaced
    InitializeParticles<<<num_blocks, num_threads_per_block>>>(particle_positions, Ni, seed, x_min, x_max); // random
    InitializeIndices<<<num_blocks, num_threads_per_block>>>(item_indices, Ni); // all -1
    checkCuda(cudaDeviceSynchronize());

    // Call CUDA kernel to find them
    BinarySearchGPU<<<num_blocks, num_threads_per_block>>>(x_grid, Nx, particle_positions, Ni, item_indices);
    /* Write Linear Search to validate */
    checkCuda(cudaDeviceSynchronize());    

    /* Validate against Linear Search */

    /* Call CPU code */


    /* Free CPU objects */

    // Free device objects
    cudaFree(x_grid);
    cudaFree(particle_positions);
    cudaFree(item_indices);
}