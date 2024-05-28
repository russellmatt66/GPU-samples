#define USE_MATH_DEFINES

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <fstream>

__device__ void BinarySearchDevice(const float *grid, const int Nx, const float item_position, int *item_indices, const int i);
__device__ void LinearSearchDevice(const float *grid, const int Nx, const float item_position, int *item_indices, const int i);
int BinarySearchHost(const float *grid, const int Nx, const float item_position);
int LinearSearchHost(const float *grid, const int Nx, const float item_position);


/*
Validate Binary Search algorithm against Linear Search
*/

/*
Device Code
*/
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
    Nx - size of grid 
    item_position - exactly where the object is located in the grid
    item_indices - the cells where the objects are in the grid
    i - the location in item_indices where the found index (j) should be placed
    */
    int low = 0;
    int high = Nx-1;
    int j = 0, counter = 0;

    while (low <= high){
        // j = floor((low + high) / 2);
        j = (low + high) / 2; // floored naturally
        counter++;
        if (grid[j] <= item_position && grid[j+1] > item_position){ // inside cell j
            item_indices[i] = j;
            break;
        }
        else if (item_position > grid[j]){ // item is to the right of cell j
            low = j+1;
        }
        else if (item_position < grid[j]){ // item is to the left of cell j
            high = j;
        }
        else if (counter >= sqrtf(Nx)){ // It's not in the grid
            item_indices[i] = -1; 
            break;
        }
    }
}

/* Locates all the particles in *item_positions, using Linear Search */
__global__ void LinearSearchGPU(const float *grid, const int Nx, const float *item_positions, const int Ni, int *item_indices){
    // Grid-stride through *item_positions and perform Linear Search to find each one
    int tnum = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for (int i = tnum; i < Ni; i += nthreads){
        LinearSearchDevice(grid, Nx, item_positions[i], item_indices, i);
    }
}

// Device implementation of Linear Search 
__device__ void LinearSearchDevice(const float *grid, const int Nx, const float item_position, int *item_indices, const int i){
    /* I know there's a way to do this concurrently, but grid-striding entangles number of threads with size of grid */
    // Serial raster scan becomes significant for large Nx
    for (int j = 0; j < Nx-1; j++){
        if (item_position >= grid[j] && item_position < grid[j+1]){
            item_indices[i] = j; // This will come in initialized to -1
        }
    }
}

// Grid-Stride through grid and set uniformly spaced values
__global__ void InitializeGrid(float *grid, const float x_min, const float dx, const int Nx){
    int tnum = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;

    for (int j = tnum; j < Nx; j += nthreads){
        grid[j] = x_min + float(j) * dx;
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

    // Linear mapping from [0.0, 1.0] to [x_min, x_max]
    float b = x_min;
    float m = x_max - b;

    for (int i = tnum; i < Ni; i += nthreads){
        float aRandomValue = static_cast<float>(curand_uniform(&state));
        particle_positions[i] = m * aRandomValue + b;
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

// Set all to false
__global__ void InitializeBool(bool *is_same, const int Ni){
    int tnum = threadIdx.x + blockDim.x * blockIdx.x; // tid or tnum is a better name for idiom
    int nthreads = blockDim.x * gridDim.x;
    
    for (int i = tnum; i < Ni; i += nthreads){
        is_same[i] = false;
    }
}

// Validate
__global__ void CompareSearchesGPU(const int* binary_indices, const int* linear_indices, const int Ni, bool* is_same, bool* passed){
    int tnum = threadIdx.x + blockDim.x * blockIdx.x; // tid or tnum is a better name for idiom
    int nthreads = blockDim.x * gridDim.x;

    for (int i = tnum; i < Ni; i += nthreads){
        if(binary_indices[i] == linear_indices[i]){
            is_same[i] = true;
        }
        else {
            *passed = false; // Comes in true
        }
    }
}

/*
Host Code
*/
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

void LinearSearchCPU(const float *grid, const int Nx, const float *item_positions, const int Ni, int *item_indices){
    for (int i = 0; i < Ni; i++){
        item_indices[i] = LinearSearchHost(grid, Nx, item_positions[i]);
    }
}

int LinearSearchHost(const float *grid, const int Nx, const float item_position){
    
    for (int j = 0; j < Nx-1; j++){
        if (item_position >= grid[j] && item_position < grid[j+1]){
            return j;
        }
    }

    return -1; // Not found
}

// Error-checking Macro from NVIDIA DLI:GSAC CUDA C/C++ course
// inline cudaError_t checkCuda(cudaError_t result)
// {
//   if (result != cudaSuccess) {
//     fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
//     assert(result == cudaSuccess);
//   }
//   return result;
// }

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


// Driver Code
int main(int argc, char* argv[]){
    /* Initialize grid, and particles inside grid */
    int Ni = std::stoi(argv[1]); // Number of items (power of two)
    int Nx = std::stoi(argv[2]); // Number of grid points (power of two)

    Ni = 1<<Ni; // lshift a binary number Ni times is equivalent to multiplying it by 2^Ni
    Nx = 1<<Nx; // " " " Nx " " " " " " " " 2^Nx

    printf("%d\n", Ni);
    printf("%d\n", Nx);

    // Device data
    float *x_grid, *particle_positions;
    int *item_indices, *item_indices_linear;

    checkCuda(cudaMallocManaged(&x_grid, Nx*sizeof(float)));
    checkCuda(cudaMallocManaged(&particle_positions, Ni*sizeof(float)));
    checkCuda(cudaMallocManaged(&item_indices, Ni*sizeof(int))); // where the particles were found with binary search
    checkCuda(cudaMallocManaged(&item_indices_linear, Ni*sizeof(int))); // where the particles were found with linear search

    float x_min = -M_PI, x_max = M_PI; /* Should I refactor this to accept these as input? */
    float dx = (x_max - x_min) / (float(Nx) - 1.0);

    printf("%lf\n", x_min);
    printf("%lf\n", x_max);

    // Set execution configuration 
    int num_blocks = 8; // GeForce GTX 960 has 8 SMs
    int num_threads_per_block = std::stoi(argv[3]);

    printf("%d\n", num_threads_per_block);

    unsigned long long seed = 1234;

    // Call CUDA kernels to initialize data
    InitializeGrid<<<num_blocks, num_threads_per_block>>>(x_grid, x_min, dx, Nx); // uniformly-spaced
    InitializeParticles<<<num_blocks, num_threads_per_block>>>(particle_positions, Ni, seed, x_min, x_max); // random
    InitializeIndices<<<num_blocks, num_threads_per_block>>>(item_indices, Ni); // all -1
    InitializeIndices<<<num_blocks, num_threads_per_block>>>(item_indices_linear, Ni); // " "
    checkCuda(cudaDeviceSynchronize());

    // Call CUDA kernel to find particles
    BinarySearchGPU<<<num_blocks, num_threads_per_block>>>(x_grid, Nx, particle_positions, Ni, item_indices);

    /* Times out for Ni = 2^21, Nx = 2^14 */
    /* Need multiple GPUs because of watchdog timer for display GPU */
    LinearSearchGPU<<<num_blocks, num_threads_per_block>>>(x_grid, Nx, particle_positions, Ni, item_indices_linear); // Validates binary search
    checkCuda(cudaDeviceSynchronize());    

    /* Validate device code using Linear Search */
    bool *is_same, *passed;

    checkCuda(cudaMallocManaged(&is_same, Ni * sizeof(bool)));
    checkCuda(cudaMallocManaged(&passed, sizeof(bool)));

    *passed = true;
    InitializeBool<<<num_blocks, num_threads_per_block>>>(is_same, Ni);
    checkCuda(cudaDeviceSynchronize());

    CompareSearchesGPU<<<num_blocks, num_threads_per_block>>>(item_indices, item_indices_linear, Ni, is_same, passed);
    checkCuda(cudaDeviceSynchronize());   

    printf("Are linear and binary search the same? %s\n", *passed ? "true" : "false");

    // Not sure I feel like doing these tbqh 
    /* Call CPU code */

    /* Validate host code using Linear Search */

    /* Free CPU objects */

    // Free device objects
    cudaFree(x_grid);
    cudaFree(particle_positions);
    cudaFree(item_indices);
    cudaFree(item_indices_linear);
    cudaFree(is_same);
    cudaFree(passed);

    return 0;
}