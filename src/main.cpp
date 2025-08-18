#include "basic_conv2d.cuh"
#include <iostream>
#include <vector>

int main()
{
    int input_rows = 4096, input_cols = 4096;
    int kernel_rows = 64, kernel_cols = 64;
    int output_rows = input_rows - kernel_rows + 1;
    int output_cols = input_cols - kernel_rows + 1;

    std::vector<float> input(static_cast<size_t>(input_rows) * input_cols, 0.0f);
    std::vector<float> kernel(static_cast<size_t>(kernel_rows) * kernel_cols, 0.0f);
    std::vector<float> output(static_cast<size_t>(output_rows) * output_cols, 0.0f);

    basicConv2D(input.data(), kernel.data(), output.data(),
                input_rows, input_cols, kernel_rows, kernel_cols);

    return 0;
}