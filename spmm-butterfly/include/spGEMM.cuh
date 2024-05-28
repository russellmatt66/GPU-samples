#ifndef SPGEMM_CUH
#define SPGEMM_CUH

#include "sparseMatrix.cuh"

__global__ void spGEMM(sparseCSRMatrix* C, sparseCSRMatrix* A, const sparseCSRMatrix* B);

#endif