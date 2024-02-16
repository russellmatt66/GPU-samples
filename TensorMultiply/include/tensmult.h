#ifndef TENSMUlT_H
#define TENSMUlT_H

#include <random>
#include <cstdint>

// row-major order
#define IDX3D(i, j, k, N) (k*N*N + i*N + j)

/* TODO - Refactor to be host code */
void InitializeTensors(float *C, float *A, float *B, const uint64_t N, const int begin, const int end){
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

	for (int i = begin; i < end; i++){
		for (int j = begin; j < end; j++){
			for (int k = begin; k < end; k++){
				A[IDX3D(i, j, k, N)] = dis(gen);
				B[IDX3D(i, j, k, N)] = dis(gen);
				C[IDX3D(i, j, k, N)] = 0.0;
			}
		}
	}
	return;
}

void HostTensorMultiply(float* C, const float* A, const float* B, const uint64_t N, const int begin, const int end){
	float sum;
	for (int i = begin; i < end; i++){
		for (int j = begin; j < end; j++){
			for (int k = begin; k < end; k++){
				sum = 0.0;
				for (int l = 0; l < N; l++){
					sum += A[IDX3D(i, j, l, N)] * B[IDX3D(l, l, k, N)];
				}
				C[IDX3D(i, j, k, N)] = sum;
			}
		}
	}
	return;
} 
#endif