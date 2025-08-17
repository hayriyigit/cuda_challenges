#include "vector_add.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

void vectorAdd(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel (256 threads per block)
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>(N / threadsPerBlock); //(N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
