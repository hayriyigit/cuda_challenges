#include "basic_conv2d.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

__global__ void convolution_2d_kernel(const float *input, const float *kernel, float *output,
                                      int input_cols, int output_rows, int output_cols, int kernel_rows, int kernel_cols)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < output_cols && row < output_rows)
    {
        float acc = 0.0f;
        for (size_t ky = 0; ky < kernel_rows; ++ky)
        {
            for (size_t kx = 0; kx < kernel_cols; ++kx)
            {
                acc += input[(row + ky) * input_cols + kx + col] * kernel[ky * kernel_cols + kx];
            }
        }
        output[row * output_cols + col] = acc;
    }
}

void basicConv2D(const float *input, const float *kernel, float *output,
                 int input_rows, int input_cols, int kernel_rows, int kernel_cols)
{
    float *dInp, *dK, *dOut;

    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_cols + 1;

    size_t bytesInp = static_cast<size_t>(input_rows) * input_cols * sizeof(float);
    size_t bytesKernel = static_cast<size_t>(kernel_rows) * kernel_cols * sizeof(float);
    size_t bytesOutput = static_cast<size_t>(output_rows) * output_cols * sizeof(float);

    cudaMalloc((void **)(&dInp), bytesInp);
    cudaMalloc((void **)(&dK), bytesKernel);
    cudaMalloc((void **)(&dOut), bytesOutput);

    cudaMemcpy(dInp, input, bytesInp, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, kernel, bytesKernel, cudaMemcpyHostToDevice);
    cudaMemcpy(dOut, output, bytesOutput, cudaMemcpyHostToDevice);

    dim3 threadSize(16, 16);
    dim3 blockSize((output_cols + threadSize.x - 1) / threadSize.x,
                   (output_rows + threadSize.y - 1) / threadSize.y);

    convolution_2d_kernel<<<blockSize, threadSize>>>(dInp, dK, dOut,
                                                     input_cols, output_rows, output_cols, kernel_rows, kernel_cols);

    cudaDeviceSynchronize();

    cudaMemcpy(output, dOut, bytesOutput, cudaMemcpyDeviceToHost);

    cudaFree(dInp);
    cudaFree(dK);
    cudaFree(dOut);
}
