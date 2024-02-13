cmake_minimum_required(VERSION 3.12)
project(Validate LANGUAGES CUDA CXX)
find_package(CUDA REQUIRED)

add_library(tensmult_lib ../include/tensmult.cu)

add_executable(validate validate.cu)

target_link_libraries(validate PRIVATE tensmult_lib)

set_target_properties(tensmult_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(validate PROPERTIES CUDA_ARCHITECTURES 75)
