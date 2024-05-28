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
			for (int k = tidz; k < N; k += zthreads){
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
(1) Use Nsight systems to observe what's going on during a run
	- Timing data is the same no matter what the inputs are
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

	// Get device attributes
	int deviceID; 
	int numberOfSMs;

	cudaGetDevice(&deviceID);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceID);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;

	/* Initialize rank-3 tensors */
	dim3 grid_dimensions(numberOfSMs * SM_mult_x, numberOfSMs * SM_mult_y, numberOfSMs * SM_mult_z);
	dim3 block_dimensions(num_threads_per_block_x, num_threads_per_block_y, num_threads_per_block_z);

	InitializeTensors<<<grid_dimensions, block_dimensions>>>(d_C, d_A, d_B, N, 1234);
	checkCuda(cudaDeviceSynchronize());

	/* Call TensorMultiply */
	cudaEventRecord(start, 0);
	TensorMultiply<<<grid_dimensions, block_dimensions>>>(d_C, d_A, d_B, N);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	// checkCuda(cudaDeviceSynchronize());

	/* Write data out to be caught by benchmarking */
	std::cout << "Size of tensors = " << pow(N,3) << std::endl;
	std::cout << "Elapsed CUDA kernel time = " << time << " ms" << std::endl;

	// Free 
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}