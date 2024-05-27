#ifndef SPARSE_MATRIX_CUH
#define SPARSE_MATRIX_CUH

// Compressed Sparse Row (CSR) format
__host__ __device__ struct sparseCSRMatrix{
    float* row_ptr; // row_ptr[num_rows] = nnz
    float* values;
    float* columns;
    int num_cols;
    int num_rows;
};

#endif