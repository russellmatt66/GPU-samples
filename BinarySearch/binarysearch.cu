/* Put binary search standalone in here to limit test */
#define USE_MATH_DEFINES

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <fstream>

__device__ void BinarySearchDevice(const float *grid, const int Nx, const float item_position, int *item_indices, const int i);

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

// Driver code
int main(int argc, char* argv[]){
    int Ni = std::stoi(argv[1]); // Number of items (power of two)
    int Nx = std::stoi(argv[2]); // Number of grid points (power of two)
    int SM_multiplier = std::atoi(argv[3]);

    Ni = 1<<Ni; // lshift a binary number Ni times is equivalent to multiplying it by 2^Ni
    Nx = 1<<Nx; // " " " Nx " " " " " " " " 2^Nx

    // Datafile for benchmarking data
    std::ofstream benchmarkFile;

    benchmarkFile.open("./benchmarking-data/N" + std::to_string(Ni) + "_Nx" + std::to_string(Nx) + ".csv");
    benchmarkFile << "nrun,num_blocks,num_threads_per_block,taukern" << std::endl;

    // Device Attributes
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    // Device data
    float *x_grid, *particle_positions;
    int *item_indices;

    checkCuda(cudaMallocManaged(&x_grid, Nx*sizeof(float)));
    checkCuda(cudaMallocManaged(&particle_positions, Ni*sizeof(float)));
    checkCuda(cudaMallocManaged(&item_indices, Ni*sizeof(int))); // where the particles were found with binary search

    float x_min = -M_PI, x_max = M_PI; /* Should I refactor this to accept these as input? */
    float dx = (x_max - x_min) / (float(Nx) - 1.0);

    // Set execution configuration 
    int num_blocks = numberOfSMs * SM_multiplier; // GeForce GTX 960 has 8 SMs
    int num_threads_per_block = std::stoi(argv[4]);
    
    unsigned long long seed = 1234;

    cudaEvent_t start_search, stop_search;
    cudaEventCreate(&start_search);
    cudaEventCreate(&stop_search);
    float time_search;

    // Call CUDA kernels to initialize data
    InitializeGrid<<<num_blocks, num_threads_per_block>>>(x_grid, x_min, dx, Nx); // uniformly-spaced
    InitializeParticles<<<num_blocks, num_threads_per_block>>>(particle_positions, Ni, seed, x_min, x_max); // random
    InitializeIndices<<<num_blocks, num_threads_per_block>>>(item_indices, Ni); // all -1
    checkCuda(cudaDeviceSynchronize());

    // Call CUDA kernel to find particles
    int num_runs = std::stoi(argv[5]);

    for (int r = 0; r < num_runs; r++){
        cudaEventRecord(start_search,0);
        BinarySearchGPU<<<num_blocks, num_threads_per_block>>>(x_grid, Nx, particle_positions, Ni, item_indices);
        cudaEventRecord(stop_search,0);
        cudaEventSynchronize(stop_search);
        cudaEventElapsedTime(&time_search, start_search, stop_search);
        benchmarkFile << r << "," << num_blocks << "," << num_threads_per_block << "," << time_search << std::endl;
    }

    // checkCuda(cudaDeviceSynchronize());

    // Free data
    cudaFree(x_grid);
    cudaFree(particle_positions);
    cudaFree(item_indices);
    benchmarkFile.close();
    return 0;
}