#include "initCSR.cuh"

__global__ void initializeButterflyMatrix(sparseCSRMatrix* A){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;

    /* Initialize CSR matrices A and B to be butterfly matrices */

    return;
}

__global__ void initializeCSRMatrix(sparseCSRMatrix* A){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    
    /* Initialize CSR matrices A and B */

    return;
}
