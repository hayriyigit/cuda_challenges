#pragma once

// Function to launch CUDA vector addition
void basicConv2D(const float* A, const float* B, float* C, int iSize, int kSize, int oSize);