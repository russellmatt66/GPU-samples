#include "spGEMM.cuh"

/* 
What is the exact state of `C` coming in? 

A priori, we can know what its dimensions are without needing to do any computation
Given the number of rows in `C`, we know how long `C->row_ptr` is: "num_rows + 1" 
Given the number of nonzero (nnz) entries in 'C' we know how long `C->values`, and `C->columns` are: "nnz"

A key piece of information is to know how many nnz entries there are in `C`.

The number of columns in `C` just tells us the upper bound on the values we will find in `C->columns`

The assumption here, then, is that the exact state of 'C' arriving at this function call is being fully allocated. 
*/
__global__ void spGEMM(sparseCSRMatrix* C, const sparseCSRMatrix* A, const sparseCSRMatrix* B){
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;

    int ridx_start = -1, next_row_idx = -1, k_A = -1; // Just give some initial value
    
    int n_col = 0;
    int B_nnz = B->row_ptr[B->num_rows + 1];

    int C_validx = -1; /* Implement the calculation of this */

    float accumulator = 0.0; // known initial state

    /* Implement GEMM between CSR butterfly matrices */
    if (tidx < C->num_rows){
        ridx_start = A->row_ptr[tidx];
        next_row_idx = A->row_ptr[tidx + 1];
        
        /* Calculate inner-product between ith-row of A ('tidx'-th) and the columns of B */
        while (n_col < B->num_cols){
            accumulator = 0.0; // clear accumulator b/w cycles

            for (int j = ridx_start; j < next_row_idx; j++){
                k_A = A->columns[j]; 
                for (int k_B = B->row_ptr[k_A]; k_B < B->row_ptr[k_A + 1]; k_B++){
                    if (B->columns[k_B] == n_col){ // Check for intersection
                        accumulator += A->values[j] * B->values[k_B];
                    }
                }
            }

            /* Calculate C_validx */
            // C->row_ptr[?] = ?;
            /* Can C->row_ptr possibly be computed upstream? C_validx can be calculated based on it, I think */
            
            C->values[C_validx] = accumulator;
            C->columns[C_validx] = n_col;
            n_col++;
        }
    }
    return;
}