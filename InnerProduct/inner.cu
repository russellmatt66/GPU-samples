#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

// Mutex for inner product
__device__ int mutex = 0;

// Should have double, and int as well
__global__ void InnerProduct(float *sum, float *a, float *b, int N){
    // Assume 1D execution configuration
    int threadNum = threadIdx.x + blockDim.x * blockIdx.x;
    int nthreads = blockDim.x * gridDim.x;
    int partial = 0;

    for (int it = threadNum; it < N; it += nthreads ){
        partial = a[it] + b[it];
    }

    while (atomicExch(&mutex, 1) != 0){
        *sum += partial; // This needs to be inside a mutex
        atomicExch(&mutex,0);
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

}