cmake_minimum_required(VERSION 3.12)
project(MatMul LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

add_executable(matmul matmul.cu)

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES 75)