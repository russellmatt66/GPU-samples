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
#define IDX3D(i, j, k, N) (k*(pow(N,2)) + i*N + j)

__global__ void TensorMultiply(float *C, const float *A, const float *B, const int N){
	// TODO - Implement these expressions
	int tidx = 0;
	int tidy = 0;
	int tidz = 0;
	int xthreads = 0;
	int ythreads = 0;
	int zthreads = 0;

	// TODO - Implement tensmult logic
	
}

int main(int argc, char* argv){
	
	return 0;
}