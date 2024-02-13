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

#include "tensmult.cuh"

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
    if (tidx < N && tidy < N){
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

// This was for zeroing out h_C b/w parallel and sequential CPU run
// void hostSetAllZero(float *C, const uint64_t N, const int begin, const int end){
// 	// row-major storage
//     for (int i = begin; i < end; i++){ 
//         for (int j = begin; j < end; j++){
//             for (int k = begin; k < end; k++){
// 				C[IDX3D(i, j, k, N)] = 0.0;
// 			}
//         }
//     }
//     return;
// }

void HostTensorMultiply(float* C, const float* A, const float* B, const uint64_t N, const int num_thread, const int stride){
	float sum;
	for (int i = num_thread; i < N; i += stride){
		for (int j = num_thread; j < N; j += stride){
			for (int k = num_thread; k < N; k += stride){
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

