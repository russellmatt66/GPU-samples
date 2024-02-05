#ifndef MATMUL_CU
#define MATMUL_CU

// #include "matrix.cu"

// row-major order
#define IDX2C(i, j, N) (((i)*(N))+(j))

// Square Matrix Multiplication
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
				sum += A[IDX2C(i, k, N)] * B[IDX2C(k, j, N)];
			}
			C[IDX2C(i, j, N)] = sum;
		}
	}
	return;
} 
#endif