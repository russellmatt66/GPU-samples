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
    // Time code using CUDA events
    cudaEvent_t start_rand, stop_rand, start_inner, stop_inner;
    float time_rand, time_inner;

    cudaEventCreate(&start_rand);
    cudaEventCreate(&stop_rand);
    cudaEventCreate(&start_inner);
    cudaEventCreate(&stop_inner);

    // Declare variables and allocate arrays
    int N = 2<<20; // left-shifting 2 twenty-times gives 2^20
    float *a, *b, *device_sum = 0;

    int size = N * sizeof(float);
    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&device_sum, sizeof(float)));

    // Initialize vectors with random data
    /* Set up execution configuration */
    int num_blocks, num_threads_per_block; 
    num_threads_per_block = 32;
    num_blocks = 32;

    cudaEventRecord(start_rand,0);
    initRandom<<<num_blocks, num_threads_per_block>>>(a, b, N);
    cudaEventRecord(stop_rand,0);
    cudaEventSynchronize(stop_rand);
    cudaEventElapsedTime(&time_rand, start_rand, stop_rand);
    // checkCuda(cudaDeviceSynchronize());

    // Call innerProduct Kernel 
    cudaEventRecord(start_inner,0);
    innerProduct<<<num_blocks, num_threads_per_block>>>(device_sum, a, b, N);
    cudaEventRecord(stop_inner,0);
    cudaEventSynchronize(stop_inner);
    cudaEventElapsedTime(&time_inner, start_inner, stop_inner);
    // checkCuda(cudaDeviceSynchronize());
    
    /* Write data out to validate */
    // Print kernel execution times
    printf("Random initialize kernel took %lf milliseconds\n", time_rand);
    printf("Inner product kernel took %lf milliseconds\n", time_inner);

    // Destroy CUDA Events
    cudaEventDestroy(start_rand);
    cudaEventDestroy(stop_rand);
    cudaEventDestroy(start_inner);
    cudaEventDestroy(stop_inner);

    // Free arrays and device_sum
    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(device_sum));
}