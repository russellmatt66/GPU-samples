/*
Kernel for benchmarking inner product computation using GeForce GTX 960 (CC 5.2)
*/

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <fstream>

// 
__global__ void innerProduct(float *sum, const float *a, const float *b, const int N){
    // Assume 1D execution configuration
    int threadNum = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    float partial = 0.0;

    for (int it = threadNum; it < N; it += nthreads ){
        partial += a[it] * b[it]; // 2 Memory Read + 1 Write
    }

    atomicAdd(sum, partial);
}

// Fill out a and b with random values according to grid-stride method 
__global__ void initRandom(float *a, float* b, const int N, const unsigned long long seed){
    int threadNum = threadIdx.x + blockDim.x * blockIdx.x; // tid or tnum is a better name for idiom
    int nthreads = blockDim.x * gridDim.x;

    // Implement device random number generation with curand 
    curandState_t state;
    if (threadNum < N){
        curand_init(seed, threadNum, 0, &state);
    }

    for (int it = threadNum; it < N; it += nthreads){
        float aRandomValue = static_cast<float>(curand_uniform(&state));
        float bRandomValue = static_cast<float>(curand_uniform(&state));
        // Maxwell 5.2 does  not support a double AtomicAdd() operation, so need to decimate random values to prevent overflow
        a[threadNum] = 0.1*aRandomValue; // 1 Memory Read + 1 Write
        b[threadNum] = 0.1*bRandomValue; // 1 Memory Read + 1 Write
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

void innerProductCPU(float *sum, const float *a, const float *b, const int N){
    for (int i = 0; i < N; i++){
        *sum += a[i] * b[i];
    }
}

// Pass execution configuration size, and array length in via command-line 
int main(int argc, char* argv[]){
    int lshift = std::stoi(argv[1]); // N = 2^(lfshift + 1)
    // Time code using CUDA events
    cudaEvent_t start_rand, stop_rand, start_inner, stop_inner;
    float time_rand, time_inner;

    cudaEventCreate(&start_rand);
    cudaEventCreate(&stop_rand);
    cudaEventCreate(&start_inner);
    cudaEventCreate(&stop_inner);

    // Declare variables and allocate arrays
    int N = 2<<lshift; // left-shifting 2 twenty-times gives 2^23
    float *a, *b, *device_sum;

    int size = N * sizeof(float);
    checkCuda(cudaMallocManaged(&a, size));
    checkCuda(cudaMallocManaged(&b, size));
    checkCuda(cudaMallocManaged(&device_sum, sizeof(float)));

    *device_sum = 0.0;

    // Set execution configuration using command-line args 
    int num_blocks, num_threads_per_block; 
    num_threads_per_block = std::stoi(argv[2]);
    num_blocks = N / num_threads_per_block;

    // Initialize vectors with random data
    unsigned long long seed = 1234;

    cudaEventRecord(start_rand,0);
    initRandom<<<num_blocks, num_threads_per_block>>>(a, b, N, seed);
    cudaEventRecord(stop_rand,0);
    cudaEventSynchronize(stop_rand);
    cudaEventElapsedTime(&time_rand, start_rand, stop_rand);
    // checkCuda(cudaDeviceSynchronize());

    // Call innerProduct kernel 
    cudaEventRecord(start_inner,0);
    innerProduct<<<num_blocks, num_threads_per_block>>>(device_sum, a, b, N);
    cudaEventRecord(stop_inner,0);
    cudaEventSynchronize(stop_inner);
    cudaEventElapsedTime(&time_inner, start_inner, stop_inner);
    // checkCuda(cudaDeviceSynchronize());


    /* Calculate CPU innerProduct and compare */
    float *host_sum;
    host_sum = (float *)malloc(sizeof(float));
    *host_sum = 0.0f;

    innerProductCPU(host_sum, a, b, N);

    float L1_norm = fabsf32(*device_sum - *host_sum);

    // printf("The inner product calculated by the CUDA kernel is %lf\n", *device_sum);
    printf("The L1 norm between the GPU and CPU inner products is %lf\n", L1_norm);

    // Print kernel execution times
    printf("initRandom kernel took %lf milliseconds\n", time_rand);
    printf("innerProduct kernel took %lf milliseconds\n", time_inner);

    // Destroy CUDA Events
    cudaEventDestroy(start_rand);
    cudaEventDestroy(stop_rand);
    cudaEventDestroy(start_inner);
    cudaEventDestroy(stop_inner);

    // Free arrays and device_sum
    checkCuda(cudaFree(a));
    checkCuda(cudaFree(b));
    checkCuda(cudaFree(device_sum));
    free(host_sum);
    // output_file.close();
}