#ifndef MATMUL_H
#define MATMUL_H

// #include <stdio.h>
#include <cstdint>	

#define IDX2D(i, j, N) (i*N + j) 
/* Host code */
void hostMatMul(float* C, const float *A, const float *B, const uint64_t N, const int begin, const int end){
    // row-major storage
    float sum;
    for (int i = begin; i < end; i++){ 
        for (int j = begin; j < end; j++){
            sum = 0.0;
            for (int k = 0; k < N; k++){
                sum += A[IDX2D(i, j, N)] * B[IDX2D(i, j, N)];
            }
            C[IDX2D(i, j, N)] = sum;
        }
    }
    return;
}

// This is for zeroing out h_C b/w parallel and sequential CPU run
// void hostSetAllZero(float *C, const uint64_t N, const int begin, const int end){
// 	// row-major storage
//     for (int i = begin; i < end; i++){ 
//         for (int j = begin; j < end; j++){
//             C[IDX2D(i, j, N)] = 0.0;
//         }
//     }
//     return;
// }
#endif