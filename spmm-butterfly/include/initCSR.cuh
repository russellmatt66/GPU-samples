#ifndef INIT_CSR_CUH
#define INIT_CSR_CUH

#include "sparseMatrix.cuh"

__host__ void initializeButterflyMatrix(int* h_rowptr, float* h_values, int* h_columns, const int i, const int L);
__global__ void initializeCSRMatrix(sparseCSRMatrix* A);

#endif