#include <cuda.h>
#include <iostream>

__global__ void gemm_kernel(int M, int N, int K, int layoutA, int layoutB, float* A, float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float valA = (layoutA == 0) ? A[row * K + k] : A[k * M + row];
            float valB = (layoutB == 0) ? B[k * N + col] : B[col * K + k];
            sum += valA * valB;
        }
        C[row * N + col] = sum;
    }
}


void launchStudentKernel(int M, int N, int K, int layoutA, int layoutB, float* A, float* B, float* C) {

    dim3 blockSize(16, 16);

    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);

    gemm_kernel<<<gridSize, blockSize>>>(M, N, K, layoutA, layoutB, A, B, C);

    cudaDeviceSynchronize();
}
