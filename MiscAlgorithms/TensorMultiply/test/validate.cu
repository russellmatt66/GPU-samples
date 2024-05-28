#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "../include/tensmult.cuh"
#include "../include/tensmult.cu"

// row-major order
#define IDX3D(i, j, k, N) (k*N*N + i*N + j)

__global__ void Validate(const float* h_C, const float* d_C, const uint64_t N, int* are_same){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    int tidz = threadIdx.z + blockDim.z * blockIdx.z;
    int xthreads = blockDim.x * gridDim.x;
    int ythreads = blockDim.y * gridDim.y;
    int zthreads = blockDim.z * gridDim.z;

    float threshold = 0.0001;
    // *are_same = 1;
    for (int i = tidx; i < N; i += xthreads){
        for (int j = tidy; j < N; j += ythreads){
            for (int k = tidz; k < N; k += zthreads){
                if (abs(h_C[IDX3D(i, j, k, N)] - d_C[IDX3D(i, j, k, N)]) > threshold){
                    printf("Not within tolerance at (i,j,k) = (%d, %d, %d)\n", i, j, k);
                    printf("d_C(i,j,k) = %f\n", d_C[IDX3D(i, j, k, N)]);
                    printf("h_C(i,j,k) = %f\n", h_C[IDX3D(i, j, k, N)]);
                    *are_same = 0;
                    break;
                }
            }
        }
    }
    return;
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

/* 
TODO 
(1) Implement code
*/
int main(int argc, char* argv[]){
    uint64_t N = atoi(argv[1]);
	int SM_mult_x = atoi(argv[2]);
    int SM_mult_y = atoi(argv[3]);
	int SM_mult_z = atoi(argv[4]);
    int num_threads_per_x = atoi(argv[5]);
    int num_threads_per_y = atoi(argv[6]);
    int num_threads_per_z = atoi(argv[7]);

    // Declare and Initialize device data
    float *A, *B, *C;

    checkCuda(cudaMalloc(&A, pow(N,3)*sizeof(float)));
    checkCuda(cudaMalloc(&B, pow(N,3)*sizeof(float)));
    checkCuda(cudaMalloc(&C, pow(N,3)*sizeof(float)));

    int deviceID;
    int numberOfSMs;

    cudaGetDevice(&deviceID);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceID);

    dim3 block_dimensions(num_threads_per_x, num_threads_per_y, num_threads_per_z);
    dim3 grid_dimensions(SM_mult_x * numberOfSMs, SM_mult_y * numberOfSMs, SM_mult_z * numberOfSMs);

    unsigned long long seed = 1234;

    InitializeTensors<<<grid_dimensions, block_dimensions>>>(C, A, B, N, seed);
    checkCuda(cudaDeviceSynchronize());

    // Migrate to host
    float *h_A, *h_B, *h_C;

    h_A = (float*)malloc(pow(N,3)*sizeof(float));
    h_B = (float*)malloc(pow(N,3)*sizeof(float));
    h_C = (float*)malloc(pow(N,3)*sizeof(float));

    // Copy A, B, and C to host 
    checkCuda(cudaMemcpy(h_A, A, pow(N,3)*sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_B, B, pow(N,3)*sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_C, C, pow(N,3)*sizeof(float), cudaMemcpyDeviceToHost));
    checkCuda(cudaDeviceSynchronize());

    // Perform tensor multiply on device
    TensorMultiply<<<grid_dimensions, block_dimensions>>>(C, A, B, N);
    // checkCuda(cudaDeviceSynchronize());

    // Perform tensor multiply on host
    int stride = 4;
    std::thread t1(HostTensorMultiply, h_C, h_A, h_B, N, 0, stride);
    std::thread t2(HostTensorMultiply, h_C, h_A, h_B, N, 1, stride);
    std::thread t3(HostTensorMultiply, h_C, h_A, h_B, N, 2, stride);
    std::thread t4(HostTensorMultiply, h_C, h_A, h_B, N, 3, stride);

    t1.join(); t2.join(); t3.join(); t4.join();
    
    checkCuda(cudaDeviceSynchronize());
    
    // Validate - Seg fault is in here
    int* are_same;
    cudaMallocManaged(&are_same, sizeof(int));
    *are_same = 1; // Page-faulting for a single int haha

    Validate<<<grid_dimensions, block_dimensions>>>(h_C, C, N, are_same);
    checkCuda(cudaDeviceSynchronize());

    if (*are_same){
        std::cout << "Host and GPU algorithms within tolerance :) " << std::endl;
    }
    else {
        std::cout << *are_same << std::endl;
        std::cout << "Host and GPU algorithms not within tolerance :( " << std::endl;
    }

    // Free
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(are_same);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}