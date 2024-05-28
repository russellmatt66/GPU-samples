#include <iostream>

#include "../include/sparseMatrix.cuh"
#include "../include/initCSR.cuh"

int main(int argc, char* argv){
	int F = 1024; // X is FxL
	int L = 1024; // W, and B_{i}, are LxL

	int deviceID;
	int numberOfSMs;

	cudaGetDevice(&deviceID);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceID);

	int threads_per_block = 1024;
	int num_blocks = numberOfSMs; // can make this an integer multiple of number of blocks 

	// Declare sparseCSRMatrix to represent Butterfly Matrix 
	sparseCSRMatrix* B;
	cudaMalloc(&B, sizeof(sparseCSRMatrix));

	std::cout << "Size of sparseCSRMatrix is: " << sizeof(sparseCSRMatrix) << " bytes" << std::endl;
	std::cout << "Size of float* is: " << sizeof(float*) << " bytes" << std::endl;
	std::cout << "Size of int is: " << sizeof(int) << " bytes" << std::endl;

	// Declare memory for pointers within sparseCSRMatrix 
	int *d_rowptr, *d_columns; 
	float *d_values;
	cudaMalloc(&d_rowptr, sizeof(int) * (L + 1));
	cudaMalloc(&d_values, sizeof(float) * (2 * L));
	cudaMalloc(&d_columns, sizeof(int) * (2 * L));

	int *h_rowptr, *h_columns;
	float *h_values;
	h_rowptr = (int*)malloc(sizeof(int) * (L + 1));
	h_values = (float*)malloc(sizeof(float) * (2 * L)); // Butterfly matrix has nnz = 2 * L
	h_columns = (int*)malloc(sizeof(int) * (2 * L)); // Butterfly matrix has nnz = 2 * L

	initializeButterflyMatrix(h_rowptr, h_values, h_columns, 0, L);

	/* Read out results from initializeButterflyMatrix */
	

	/* Free all data */
	// Device
	cudaFree(B);
	cudaFree(d_rowptr);
	cudaFree(d_values);
	cudaFree(d_columns);
	
	// Host
	free(h_rowptr);
	free(h_values);
	free(h_columns);
	return 0;
}