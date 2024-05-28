#ifndef TENSMULT_CUH
#define TENSMULT_CUH

#include <cstdint>

__global__ void InitializeTensors(float*, float*, float*, const uint64_t, const unsigned long long);

__global__ void TensorMultiply(float*, const float*, const float*, const uint64_t);

// void hostSetAllZero(float*, const uint64_t, const int, const int);

void HostTensorMultiply(float*, const float*, const float*, const uint64_t, const int, const int);

#endif