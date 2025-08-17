cmake_minimum_required(VERSION 3.10)
project(VectorAddCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 75)


add_executable(vector_add
    src/main.cpp
    src/basic_conv2d.cu
)

set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
