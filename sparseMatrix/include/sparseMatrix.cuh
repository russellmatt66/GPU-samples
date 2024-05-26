#ifndef SPARSE_MATRIX_CUH
#define SPARSE_MATRIX_CUH

__host__ __device__ struct sparseCSRMatrix{
    float* row_ptr;
    float* values;
    float* columns;
};

#endif