#include "basic_conv2d.cuh"
#include <iostream>
#include <vector>

int main()
{
    int iSize = 6, kSize = 3;
    int oSize = iSize - kSize + 1;

    std::vector<float> input(static_cast<size_t>(iSize) * iSize, 0.0f);
    std::vector<float> kernel(static_cast<size_t>(kSize) * kSize, 0.0f);
    std::vector<float> output(static_cast<size_t>(oSize) * oSize, 0.0f);

    for (size_t i = 0; i < iSize; ++i)
    {
        for (size_t j = 0; j < iSize; ++j)
        {
            if (j < 3)
            {
                input[i * iSize + j] = 10.0f;
            }
        }
    }

    for (size_t i = 0; i < kSize; ++i)
    {
        for (size_t j = 0; j < kSize; ++j)
        {
            if (j == 0)
            {
                kernel[i * kSize + j] = 1.0f;
            }
            if (j == 2)
            {
                kernel[i * kSize + j] = -1.0f;
            }
        }
    }

    basicConv2D(input.data(), kernel.data(), output.data(), iSize, kSize, oSize);

    for (size_t i = 0; i < oSize; ++i)
    {
        for (size_t j = 0; j < oSize; ++j)
        {

            std::cout << output[i * oSize + j] << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}