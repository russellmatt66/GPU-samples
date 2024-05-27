#include "initCSR.cuh"

/* Initialize all values to 2.0 */
__host__ void initializeButterflyMatrix(int* h_rowptr, float* h_values, int* h_columns, const int i, const int L){

    for (int k = 0; k < L; k++){
        h_rowptr[k] = 2 * k;
    }
    h_rowptr[L] = 2 * L;

    h_values[0] = 2.0;
    h_columns[0] = 0;
    int ridx = 0; // track which row the values are in
    for (int k = 1; k < 2 * L; k++){
        h_values[k] = 2.0;

        if (h_rowptr[ridx] == k){ // first entry in the row
            h_columns[k] = ridx < ridx ^ i ? ridx : ridx ^ i;
        }
        else { // second entry in the row
            h_columns[k] = ridx > ridx ^ i ? ridx : ridx ^ i;
        }
        
        if (h_rowptr[ridx] < k) {
            ridx++;
        }
    }
    return;
}

__global__ void initializeCSRMatrix(sparseCSRMatrix* A){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    
    /* Initialize CSR matrices A and B */

    return;
}
