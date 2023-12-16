/*
Kernel for benchmarking inner product computation using GeForce GTX 960 
*/

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

// Should have double, and 16-bit (?) kernels as well ("?" = if possible)
__global__ void innerProduct(float *sum, float *a, float *b, int N){
    // Assume 1D execution configuration
    int threadNum = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    int partial = 0;

    for (int it = threadNum; it < N; it += nthreads ){
        partial += a[it] * b[it];
    }

    atomicAdd(sum, partial);
}

// Fill out a and b with random values according to grid-stride method 
__global__ void initRandom(float *a, float* b, int N){
    int threadNum = threadIdx.x + blockDim.x * blockIdx.x; // tid or tnum is a better name for idiom
    int nthreads = blockDim.x * gridDim.x;

    /* Implement device random number generation with curand */
    curandState_t state;
    unsigned long long seed = 1234;
    if (threadNum < N){
        curand_init(seed, threadNum, 0, &state);
    }

    for (int it = threadNum; it < N; it += nthreads){
        float aRandomValue = static_cast<float>(curand_uniform(&state));
        a[threadNum] = aRandomValue;
        float bRandomValue = static_cast<float>(curand_uniform(&state));
        b[threadNum] = bRandomValue;
    }
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

int main(){
    // Declare variables and allocate arrays
    int N = 2<<20; // left-shifting 2 twenty-times gives 2^20
    float *a, *b, *sum = 0;

    int size = N * sizeof(float);
    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));

    // Initialize vectors with random data
    /* Set up execution configuration */
    int num_blocks, num_threads_per_block; 
    num_threads_per_block = 32;
    num_blocks = 1;

    initRandom<<<num_blocks, num_threads_per_block>>>(a, b, N);
    checkCuda(cudaDeviceSynchronize());

    /* Copy from host to device */
    checkCuda(cudaMemcpy());

    // Call innerProduct Kernel 
    /* There's an illegal memory access occuring in here */
    // innerProduct<<<num_blocks, num_threads_per_block>>>(sum, a, b, N);
    // checkCuda(cudaDeviceSynchronize());
    
    // Free arrays
    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
}