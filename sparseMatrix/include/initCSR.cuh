#ifndef INIT_CSR_CUH
#define INIT_CSR_CUH

#include "sparseMatrix.cuh"

__global__ void initializeButterflyMatrix(sparseCSRMatrix* A);
__global__ void initializeCSRMatrix(sparseCSRMatrix* A);

#endif