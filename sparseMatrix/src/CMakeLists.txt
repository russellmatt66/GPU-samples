cmake_minimum_required(VERSION 3.12)
project(butterflyMM LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

add_library(init_CSR_lib ../include/initCSR.cu)
add_library(spGEMM_lib ../include/spGEMM.cu)

add_executable(butterfly-mm main.cu)

target_include_directories(butterfly-mm PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(butterfly-mm PRIVATE init_CSR_lib)
target_link_libraries(butterfly-mm PRIVATE spGEMM_lib)

set_target_properties(init_CSR_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(spGEMM_lib PROPERTIES CUDA_ARCHITECTURES 75)
set_target_properties(butterfly-mm PROPERTIES CUDA_ARCHITECTURES 75)