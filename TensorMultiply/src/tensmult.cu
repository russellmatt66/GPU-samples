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
#include <cstdio>

#include "../include/tensmult.cuh"

// #include "../include/tensmult.cu"

// row-major order
#define IDX3D(i, j, k, N) (k*N*N + i*N + j)

__global__ void InitializeTensors(float *C, float *A, float *B, const uint64_t N, const unsigned long long seed){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y; 
	int tidz = threadIdx.z + blockIdx.z * blockDim.z;
	int xthreads = gridDim.x * blockDim.x;
	int ythreads = gridDim.y * blockDim.y; 
	int zthreads = gridDim.z * blockDim.z;

	// Generate random number b/w [0.0, 1.0]
    curandState_t state;
    if (tidx < N && tidy < N && tidz < N){
        curand_init(seed, tidx, 0, &state);
    }

	for (int i = tidx; i < N; i += xthreads){
		for (int j = tidy; j < N; j += ythreads){
			for (int k = tidz; k < N; j += zthreads){
				A[IDX3D(i, j, k, N)] = static_cast<float>(curand_uniform(&state));
				B[IDX3D(i, j, k, N)] = static_cast<float>(curand_uniform(&state));
				C[IDX3D(i, j, k, N)] = 0.0;
			}
		}
	}
	return;
}

__global__ void TensorMultiply(float *C, const float *A, const float *B, const uint64_t N){
	int tidx = threadIdx.x + blockDim.x * blockIdx.x;
	int tidy = threadIdx.y + blockDim.y * blockIdx.y;
	int tidz = threadIdx.z + blockDim.z * blockIdx.z;
	int xthreads = gridDim.x * blockDim.x;
	int ythreads = gridDim.y * blockDim.y;
	int zthreads = gridDim.z * blockDim.z;

	float sum;
	for (int i = tidx; i < N; i += xthreads){
		for (int j = tidy; j < N; j += ythreads){
			for (int k = tidz; k < N; k += zthreads){
				sum = 0.0;
				for (int l = 0; l < N; l++){
					sum += A[IDX3D(i, j, l, N)] * B[IDX3D(l, l, k, N)];
				}
				C[IDX3D(i, j, k, N)] = sum;
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

*/
int main(int argc, char* argv[]){
	uint64_t N = atoll(argv[1]);
	int SM_mult_x = atoi(argv[2]);
	int SM_mult_y = atoi(argv[3]);
	int SM_mult_z = atoi(argv[4]);
	int num_threads_per_block_x = atoi(argv[5]);
	int num_threads_per_block_y = atoi(argv[6]);
	int num_threads_per_block_z = atoi(argv[7]);
	
	uint64_t requested_memory = N*N*N*sizeof(float);

	float *d_C, *d_A, *d_B; 

	checkCuda(cudaMalloc(&d_C, requested_memory));
	checkCuda(cudaMalloc(&d_A, requested_memory));
	checkCuda(cudaMalloc(&d_B, requested_memory));

	/* Initialize rank-3 tensors */

	/* Call TensorMultiply */

	/* Write data out to be caught by benchmarking */

	// Free 
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}