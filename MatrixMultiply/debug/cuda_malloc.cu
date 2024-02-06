// Problem:
// Using cudaMallocManaged() for the matrices results in some weird behavior:
// executable can run with far larger matrices than it should be able to
// Solution:
// Use cudaMalloc, and do migrations manually 
#include <cstdlib>
#include <cstdint>
#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>

#define IDX2D(i, j, N) (i * N + j) // row-major order

__global__ void setAllZero(float *A, float *B, float *C, uint64_t N){
    int tidx = threadIdx.x + blockDim.x * gridDim.x;
    int tidy = threadIdx.y + blockDim.y * gridDim.y;
    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;

    for (int i = tidx; i < N; i += xthreads){
        for (int j = tidy; j < N; j += ythreads){
            A[IDX2D(i, j, N)] = 0.0;
            B[IDX2D(i, j, N)] = 0.0;
            C[IDX2D(i, j, N)] = 0.0;
        }
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
    // int N = atoi(argv[1]);
    uint64_t N = atoll(argv[1]);

    std::cout << "Size of N: " << sizeof(N) << std::endl;

    float *A, *B, *C; // flattened arrays because that is easiest with CUDA  

    uint64_t requested_matrix_memory = N*N*sizeof(float);
    
    std::cout << "Size of requested memory for single matrix is: " << requested_matrix_memory << " bytes" << std::endl; 

    // The problem seems to be with `cudaMallocManaged` - it does not allocate memory to the device by default
    // checkCuda(cudaMallocManaged(&A, 0); `size = 0` didn't yield an error like docs say 
    // checkCuda(cudaMallocManaged(&A, requested_matrix_memory)); 
    // checkCuda(cudaGetLastError());
	// checkCuda(cudaMallocManaged(&B, requested_matrix_memory)); 
    // checkCuda(cudaGetLastError());
	// checkCuda(cudaMallocManaged(&C, requested_matrix_memory));
    // checkCuda(cudaGetLastError());
    // For some reason the below doesn't work, hence why I tried declaring the requested memory separately 
    // checkCuda(cudaMallocManaged(&A, ((uint64_t)N)*((uint64_t)N)*sizeof(float))); // cast b/c of integer overflow 
	// checkCuda(cudaMallocManaged(&B, ((uint64_t)N)*((uint64_t)N)*sizeof(float))); // pow(N,2) not behaving
	// checkCuda(cudaMallocManaged(&C, ((uint64_t)N)*((uint64_t)N)*sizeof(float)));
    
    checkCuda(cudaMalloc(&A, requested_matrix_memory));
    checkCuda(cudaMalloc(&B, requested_matrix_memory));
    checkCuda(cudaMalloc(&C, requested_matrix_memory));

    size_t free_bytes, total_bytes;

    checkCuda(cudaMemGetInfo(&free_bytes, &total_bytes));

    std::cout << "Free memory on GPU: " << free_bytes << " bytes" << std::endl;
    std::cout << "Total memory on GPU: " << total_bytes << " bytes" << std::endl;

    // Do something just so the compiler doesn't optimize it away
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    dim3 block_dimensions (32, 32);
    dim3 grid_dimensions (numberOfSMs, numberOfSMs); 

    setAllZero<<<block_dimensions, grid_dimensions>>>(A, B, C, N);

    std::cout << "Free memory on GPU: " << free_bytes << " bytes" << std::endl;
    std::cout << "Total memory on GPU: " << total_bytes << " bytes" << std::endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}