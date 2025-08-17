#include "matrix_multiplication.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

__global__ void matrix_multiplication_kernel(const float *A, const float *B, float *C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K)
    {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i)
        {
            sum += A[row * N + i] * B[i*K + col];
        }
        C[row * K + col] = sum;
        
    }
}

void multiplyMatrices(const float *A, const float *B, float *C, int M, int N, int K)
{
    float *d_A, *d_B, *d_C;
    size_t size_A = static_cast<size_t>(M) * sizeof(float) * N;
    size_t size_B = static_cast<size_t>(N) * sizeof(float) * K;
    size_t size_C = static_cast<size_t>(M) * sizeof(float) * K;

    cudaMalloc((void **)&d_A, size_A);
    cudaMalloc((void **)&d_B, size_B);
    cudaMalloc((void **)&d_C, size_C);
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size_C, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}