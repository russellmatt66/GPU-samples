#ifndef SPARSE_MATRIX_CUH
#define SPARSE_MATRIX_CUH

// Compressed Sparse Row (CSR) format
struct sparseCSRMatrix{
    int* row_ptr; // row_ptr[num_rows] = nnz
    float* values;
    int* columns;
    int num_cols;
    int num_rows;
};

#endif