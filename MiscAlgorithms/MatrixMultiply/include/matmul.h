#ifndef MATMUL_H
#define MATMUL_H

// #include <stdio.h>
#include <cstdint> 
#include <random>

#define IDX2D(i, j, N) (i*N + j) 
/* Host code */
void initMatrices(float *C, float *A, float *B, const uint64_t N, const int begin, const int end){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = begin; i < end; i++){
        for (int j = begin; j < end; j++){
            A[IDX2D(i, j, N)] = dis(gen);
            B[IDX2D(i, j, N)] = dis(gen);
            C[IDX2D(i, j, N)] = 0.0;
        }
    }
    return;
}

void hostMatMul(float* C, const float *A, const float *B, const uint64_t N, const int begin, const int end){
    // row-major storage
    float sum;
    for (int i = begin; i < end; i++){ 
        for (int j = begin; j < end; j++){
            sum = 0.0;
            for (size_t k = 0; k < N; k++){
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