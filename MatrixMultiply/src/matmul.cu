// #include "../include/matrix.cu"
// #include "../include/matmul.cu"

#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <cstdlib>
#include <cstdint>	
// #include <filesystem>
// #include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

// row-major order
#define IDX2D(i, j, N) (((i)*(N))+(j))

/* Device code */
__global__ void InitializeMatrices(float *C, float *A, float *B, const uint64_t N, const unsigned long long seed){
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

__global__ void MatMul(float *C, const float *A, const float *B, const uint64_t N){ 
	int tidx = threadIdx.x + blockIdx.x * blockDim.x;
	int tidy = threadIdx.y + blockIdx.y * blockDim.y; 
	int xthreads = gridDim.x * blockDim.x;
	int ythreads = gridDim.y * blockDim.y; 

	float sum;
	for (int i = tidx; i < N; i += xthreads){
		for (int j = tidy; j < N; j += ythreads){
			sum = 0.0;
			for (int k = 0; k < N; k++){
				sum += A[IDX2D(i, k, N)] * B[IDX2D(k, j, N)];
			}
			C[IDX2D(i, j, N)] = sum;
		}
	}
	return;
} 

/* Host code */
void hostMatMul(float* C, const float *A, const float *B, const uint64_t N, const int begin, const int end){
    // row-major storage
    float sum;
    for (int i = begin; i < end; i++){ 
        for (int j = begin; j < end; j++){
            sum = 0.0;
            for (int k = 0; k < N; k++){
                sum += A[IDX2D(i, j, N)] * B[IDX2D(i, j, N)];
            }
            C[IDX2D(i, j, N)] = sum;
        }
    }
    return;
}

// This is for zeroing out h_C b/w parallel and sequential CPU run
void hostSetAllZero(float *C, const uint64_t N, const int begin, const int end){
	// row-major storage
    for (int i = begin; i < end; i++){ 
        for (int j = begin; j < end; j++){
            C[IDX2D(i, j, N)] = 0.0;
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
(1) Cleanup comments
*/  
int main(int argc, char* argv[]){
	// Accept arguments 
	uint64_t N = atoll(argv[1]);
	int SM_multiplier_x = atoi(argv[2]); // used for changing number of blocks
	int SM_multiplier_y = atoi(argv[3]);
	int num_threads_per_block_x = atoi(argv[4]);
	int num_threads_per_block_y = atoi(argv[5]);

	// Allocate device matrices
	float *A, *B, *C; // flattened arrays because that is easiest with CUDA
    
	uint64_t requested_matrix_memory = N*N*sizeof(float);

	checkCuda(cudaMalloc(&A, requested_matrix_memory));
	checkCuda(cudaMalloc(&B, requested_matrix_memory));
	checkCuda(cudaMalloc(&C, requested_matrix_memory));

	std::cout << "Size of matrices is = " << pow(N,2) << std::endl;

    // Get device attributes 
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

	std::cout << "Number of SMs on device = " << numberOfSMs << std::endl;

	// Define execution configuration
	dim3 block_dimensions(num_threads_per_block_x, num_threads_per_block_y, 1);
	dim3 grid_dimensions(numberOfSMs * SM_multiplier_x, numberOfSMs * SM_multiplier_y, 1);

	// Set up timer
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time;

	// Initialize Matrices
	InitializeMatrices<<<block_dimensions, grid_dimensions>>>(C, A, B, N, 1234); // Magic number at the end is seed for rng
	checkCuda(cudaDeviceSynchronize());

	// Allocate host matrices
	float *h_A, *h_B, *h_C;

	h_A = (float*)malloc(requested_matrix_memory);
	h_B = (float*)malloc(requested_matrix_memory);
	h_C = (float*)malloc(requested_matrix_memory);

	checkCuda(cudaMemcpy(h_A, A, requested_matrix_memory, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_B, B, requested_matrix_memory, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(h_C, C, requested_matrix_memory, cudaMemcpyDeviceToHost));

	// Perform Matrix Multiplication
	// GPU kernel, and CPU function, are validated in `../test/validate_matmul.cu`
	// Device
	cudaEventRecord(start, 0);
	MatMul<<<grid_dimensions, block_dimensions>>>(C, A, B, N);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	std::cout << "Elapsed CUDA kernel time is = " << time << " ms" << std::endl;

	// Host Code
	// Parallel
	auto start_host = std::chrono::high_resolution_clock::now();
	std::thread t1(hostMatMul, h_C, h_A, h_B, N, 0, N/8);
	std::thread t2(hostMatMul, h_C, h_A, h_B, N, N/8, N/4);
	std::thread t3(hostMatMul, h_C, h_A, h_B, N, N/4, 3*N/8);
	std::thread t4(hostMatMul, h_C, h_A, h_B, N, 3*N/8, N/2);
	std::thread t5(hostMatMul, h_C, h_A, h_B, N, N/2, 5*N/8);
	std::thread t6(hostMatMul, h_C, h_A, h_B, N, 5*N/8, 3*N/4);
	std::thread t7(hostMatMul, h_C, h_A, h_B, N, 3*N/4, 7*N/8);
	std::thread t8(hostMatMul, h_C, h_A, h_B, N, 7*N/8, N);

	t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join(); t7.join(); t8.join();

	auto stop_host = std::chrono::high_resolution_clock::now();
	auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_host - start_host).count();

	// std::cout << "Elapsed multi-threaded C++ time is: " << elapsed_time << " ms" << std::endl;
	printf("Elapsed multi-threaded C++ time is = %ld us\n", elapsed_time);
	std::cout << "Number of CPU cores = " << 8 << std::endl; 

	// Free data
	cudaFree(A);
	cudaFree(B);
	cudaFree(C);
	free(h_A);
	free(h_B);
	free(h_C);
	return 0;
}