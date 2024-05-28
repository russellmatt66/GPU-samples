#include <iostream>
#include <thread>
#include <chrono>
#include <stdlib.h>

#include "../include/matmul.h"

// TODO - Move CPU benchmarking code into here
int main(int argc, char* argv[]){
	uint64_t N = atoll(argv[1]);
	uint64_t requested_matrix_memory = N*N*sizeof(float);

	// Allocate host matrices
	float *h_A, *h_B, *h_C;

	h_A = (float*)malloc(requested_matrix_memory);
	h_B = (float*)malloc(requested_matrix_memory);
	h_C = (float*)malloc(requested_matrix_memory);

	/* TODO - Initialize matrices */
	std::thread i1(initMatrices, h_C, h_A, h_B, N, 0, N/8);
	std::thread i2(initMatrices, h_C, h_A, h_B, N, N/8, N/4);
	std::thread i3(initMatrices, h_C, h_A, h_B, N, N/4, 3*N/8);
	std::thread i4(initMatrices, h_C, h_A, h_B, N, 3*N/8, N/2);
	std::thread i5(initMatrices, h_C, h_A, h_B, N, N/2, 5*N/8);
	std::thread i6(initMatrices, h_C, h_A, h_B, N, 5*N/8, 3*N/4);
	std::thread i7(initMatrices, h_C, h_A, h_B, N, 3*N/4, 7*N/8);
	std::thread i8(initMatrices, h_C, h_A, h_B, N, 7*N/8, N);

	i1.join(); i2.join(); i3.join(); i4.join(); i5.join(); i6.join(); i7.join(); i8.join();

    // Host Code
	// Parallel
	auto start_host = std::chrono::high_resolution_clock::now();
	std::thread t1(hostMatMul, h_C, h_A, h_B, N, 0, N/8);
	std::thread t2(hostMatMul, h_C, h_A, h_B, N, N/8, N/4);
	std::thread t3(hostMatMul, h_C, h_A, h_B, N, N/4, 3*N/8);
	std::thread t4(hostMatMul, h_C, h_A, h_B, N, 3*N/8, N/2);
	std::thread t5(hostMatMul, h_C, h_A, h_B, N, N/2, 5*N/8);
	std::thread t6(hostMatMul, h_C, h_A, h_B, N, 5*N/8, 3*N/4);
	std::thread t7(hostMatMul, h_C, h_A, h_B, N, 3*N/4, 7*N/8);
	std::thread t8(hostMatMul, h_C, h_A, h_B, N, 7*N/8, N);

	t1.join(); t2.join(); t3.join(); t4.join(); t5.join(); t6.join(); t7.join(); t8.join();
	auto stop_host = std::chrono::high_resolution_clock::now();
	auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_host - start_host).count();
	printf("Elapsed multi-threaded C++ time is = %ld us\n", elapsed_time);

	// std::cout << "Elapsed multi-threaded C++ time is: " << elapsed_time << " ms" << std::endl;
	std::cout << "Number of CPU cores = " << 8 << std::endl; 

	free(h_A);
	free(h_B);
	free(h_C);
    return 0;
}