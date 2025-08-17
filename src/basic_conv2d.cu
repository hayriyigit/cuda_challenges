#include "basic_conv2d.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

__global__ void basic_conv2d_kernel(const float *A, const float *B, float *C, int iSize, int kSize, int oSize)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < oSize && col < oSize)
    {
        float sum = 0.0f;
        for (size_t ky = 0; ky < kSize; ++ky)
        {
            for (size_t kx = 0; kx < kSize; ++kx)
            {
                sum += B[ky * kSize + kx] * A[(row + ky) * iSize + col + kx];
            }
            C[row * oSize + col] = sum;
        }
    }
}

void basicConv2D(const float *A, const float *B, float *C, int iSize, int kSize, int oSize)
{
    float *dIn, *dK, *dOut;

    size_t bytesInput = static_cast<size_t>(iSize) * iSize * sizeof(float);
    size_t bytesKernel = static_cast<size_t>(kSize) * kSize * sizeof(float);
    size_t bytesOutput = static_cast<size_t>(oSize) * oSize * sizeof(float);

    cudaMalloc((void **)(&dIn), bytesInput);
    cudaMalloc((void **)(&dK), bytesKernel);
    cudaMalloc((void **)(&dOut), bytesOutput);

    cudaMemcpy(dIn, A, bytesInput, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, B, bytesKernel, cudaMemcpyHostToDevice);
    cudaMemcpy(dOut, C, bytesOutput, cudaMemcpyHostToDevice);

    dim3 threadSize(16, 16);
    dim3 blockSize((oSize + threadSize.x - 1) / threadSize.x,
                   (oSize + threadSize.y - 1) / threadSize.y);

    basic_conv2d_kernel<<<blockSize, threadSize>>>(dIn, dK, dOut, iSize, kSize, oSize);
    cudaDeviceSynchronize();

    cudaMemcpy(C, dOut, bytesOutput, cudaMemcpyDeviceToHost);

    cudaFree(dIn);
    cudaFree(dK);
    cudaFree(dOut);
}
