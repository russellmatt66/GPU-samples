cmake_minimum_required(VERSION 3.12)
project(TensorMultiply CUDA CXX)
find_package(CUDA REQUIRED)

add_executable(gpu_tensmult tensmult.cu)
# add_executable(cpu_tensmult tensmult.cpp)

set_target_properties(gpu_tensmult PROPERTIES CUDA_ARCHITECTURES 75)