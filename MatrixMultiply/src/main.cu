// #include "../include/matrix.cu"
// #include "../include/matmul.cu"

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

// row-major order
#define IDX2D(i, j, N) (((i)*(N))+(j))

__global__ void InitializeMatrices(float *C, float *A, float *B, const int N, const unsigned long long seed){
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y; 
	int xthreads = gridDim.x * blockDim.x;
	int ythreads = gridDim.y * blockDim.y; 

	// Generate random number b/w [0.0, 1.0]
    curandState_t state;
    if (tidx < N && tidy < N){
        curand_init(seed, tidx, 0, &state);
    }

	for (int i = tidx; i < N; i += xthreads){
		for (int j = tidy; j < N; j += ythreads){
			A[IDX2D(i, j, N)] = static_cast<float>(curand_uniform(&state));
			B[IDX2D(i, j, N)] = static_cast<float>(curand_uniform(&state));
			C[IDX2D(i, j, N)] = 0.0;
		}
	}
	return;
}

__global__ void MatMul(float *C, const float *A, const float *B, const int N){ 
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y; 
	int xthreads = gridDim.x * blockDim.x;
	int ythreads = gridDim.y * blockDim.y; 

	int sum;
	for (int i = tidx; i < N; i += xthreads){
		for (int j = tidy; j < N; j += ythreads){
			sum = 0;
			for (int k = 0; k < N; k++){
				sum += A[IDX2D(i, k, N)] * B[IDX2D(k, j, N)];
			}
			C[IDX2D(i, j, N)] = sum;
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

int main(int argc, char* argv[]){
	// Accept arguments 
	int N = atoi(argv[1]); // length of matrix side
	int SM_multiplier_x = atoi(argv[2]); // used for changing number of blocks
	int SM_multiplier_y = atoi(argv[3]);
	int num_threads_per_block_x = atoi(argv[4]);
	int num_threads_per_block_y = atoi(argv[5]);

	// Allocate Matrices on Device
	float *A, *B, *C; // flattened arrays because that is easiest with CUDA

	checkCuda(cudaMallocManaged(&A, pow(N,2)*sizeof(float)));
	checkCuda(cudaMallocManaged(&B, pow(N,2)*sizeof(float)));
	checkCuda(cudaMallocManaged(&C, pow(N,2)*sizeof(float)));

	std::cout << "Size of matrices is: " << pow(N,2) << std::endl;

    // Get device attributes 
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	// Define execution configuration
	dim3 block_dimensions(num_threads_per_block_x, num_threads_per_block_y);
	dim3 grid_dimensions(numberOfSMs * SM_multiplier_x, numberOfSMs * SM_multiplier_y);

	// Initialize Matrices
	InitializeMatrices<<<block_dimensions, grid_dimensions>>>(C, A, B, N, 1234); // Magic number at the end is seed for rng
	
	// Perform Matrix Multiplication
	MatMul<<<block_dimensions, grid_dimensions>>>(C, A, B, N);

	return 0;
}